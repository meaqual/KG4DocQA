import json
import os
from openai import OpenAI

# Initialize OpenAI client for Qwen
host = "http://120.25.194.125:2105/v1"
apikey = "xxx"
model = "qwen_docqa"

client = OpenAI(
    base_url=host,
    api_key=apikey,
    timeout=300.0
)

def load_extraction_results(file_path):
    """Load the extraction results JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_classes_with_chunk_id(data):
    """
    Extract all classes from the 'extracted' field and add chunk_id to each example.
    Returns a dictionary where keys are class names and values are lists of examples.
    """
    all_classes = {}
    
    for chunk in data:
        chunk_id = chunk.get("chunk_id", "")
        extracted = chunk.get("extracted", {})
        
        for class_name, examples in extracted.items():
            if class_name not in all_classes:
                all_classes[class_name] = []
            
            for example in examples:
                # Skip empty examples
                if not example or (isinstance(example, dict) and not any(v for v in example.values() if v)):
                    continue
                
                # Add chunk_id to the example
                example_with_chunk = example.copy() if isinstance(example, dict) else {"value": example}
                example_with_chunk["chunk_id"] = chunk_id
                all_classes[class_name].append(example_with_chunk)
    
    return all_classes

def group_by_class_and_name(all_classes):
    """
    Group examples by (class, name) pairs for merging.
    Returns a dictionary where keys are (class_name, name) tuples and values are lists of examples.
    """
    grouped = {}
    
    for class_name, examples in all_classes.items():
        for example in examples:
            name = example.get("name", None)
            if name is None:
                # If no name field, use a unique key
                key = (class_name, f"_unnamed_{id(example)}")
            else:
                key = (class_name, name)
            
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(example)
    
    return grouped

def merge_examples_with_qwen(examples, class_name):
    """
    Use Qwen to merge multiple examples with the same class and name.
    """
    if len(examples) == 1:
        return examples[0]
    
    # Prepare the prompt for merging
    examples_json = json.dumps(examples, ensure_ascii=False, indent=2)
    
    prompt = f"""你是一个EDA文档知识合并专家。以下是多个关于同一个{class_name}的提取结果，它们具有相同的class和name。
请将这些结果合并成一个完整的条目，要求：
1. 保持原有的字段结构不变
2. 合并时选择最完整、最详细的信息
3. 如果某个字段在不同条目中有不同的值，选择最详细或最准确的那个
4. chunk_id字段应该合并为一个列表，包含所有来源的chunk_id
5. 如果有冲突的信息，优先保留更详细的描述
6. 返回合并后的JSON对象

需要合并的条目：
{examples_json}

请直接返回合并后的JSON对象，不要包含任何其他文字说明。"""

    try:
        response = client.chat.completions.create(
            model= model,
            messages=[
                {"role": "system", "content": "你是一个EDA文档知识合并专家，专门负责合并重复的知识条目。只返回JSON格式的结果。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=4096
        )
        
        result_text = response.choices[0].message.content.strip()
        
        # Try to extract JSON from the response
        # Handle cases where the response might have markdown code blocks
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0].strip()
        elif "```" in result_text:
            result_text = result_text.split("```")[1].split("```")[0].strip()
        
        # Remove thinking tags if present
        if "<think>" in result_text:
            result_text = result_text.split("</think>")[-1].strip()
        
        merged = json.loads(result_text)
        return merged
        
    except Exception as e:
        print(f"Error merging examples for {class_name}: {e}")
        # Fallback: manual merge
        return manual_merge(examples)

def manual_merge(examples):
    """
    Manually merge examples when Qwen fails.
    """
    if len(examples) == 1:
        return examples[0]
    
    merged = {}
    chunk_ids = []
    
    for example in examples:
        for key, value in example.items():
            if key == "chunk_id":
                chunk_ids.append(value)
            elif key not in merged or (value and not merged[key]):
                merged[key] = value
            elif value and merged[key] and len(str(value)) > len(str(merged[key])):
                # Keep the longer/more detailed value
                merged[key] = value
    
    merged["chunk_id"] = chunk_ids if len(chunk_ids) > 1 else (chunk_ids[0] if chunk_ids else None)
    return merged

def process_and_merge(input_file, output_dir):
    """
    Main function to process the extraction results and merge duplicates.
    """
    # Load data
    print("Loading extraction results...")
    data = load_extraction_results(input_file)
    print(f"Loaded {len(data)} chunks")
    
    # Extract classes with chunk_id
    print("Extracting classes with chunk_id...")
    all_classes = extract_classes_with_chunk_id(data)
    
    # Save intermediate result (before merging)
    os.makedirs(output_dir, exist_ok=True)
    intermediate_file = os.path.join(output_dir, "extracted_classes_with_chunk_id.json")
    with open(intermediate_file, 'w', encoding='utf-8') as f:
        json.dump(all_classes, f, ensure_ascii=False, indent=2)
    print(f"Saved intermediate result to {intermediate_file}")
    
    # Print statistics
    print("\nClass statistics (before merging):")
    for class_name, examples in all_classes.items():
        print(f"  {class_name}: {len(examples)} examples")
    
    # Group by (class, name)
    print("\nGrouping by (class, name)...")
    grouped = group_by_class_and_name(all_classes)
    
    # Find duplicates (groups with more than 1 example)
    duplicates = {k: v for k, v in grouped.items() if len(v) > 1}
    print(f"Found {len(duplicates)} groups with duplicates")
    
    # Merge duplicates using Qwen
    print("\nMerging duplicates...")
    merged_classes = {}
    
    for (class_name, name), examples in grouped.items():
        if class_name not in merged_classes:
            merged_classes[class_name] = []
        
        if len(examples) > 1:
            print(f"  Merging {len(examples)} examples for {class_name}/{name}...")
            merged = merge_examples_with_qwen(examples, class_name)
        else:
            merged = examples[0]
        
        merged_classes[class_name].append(merged)
    
    # Save merged result
    merged_file = os.path.join(output_dir, "merged_classes.json")
    with open(merged_file, 'w', encoding='utf-8') as f:
        json.dump(merged_classes, f, ensure_ascii=False, indent=2)
    print(f"\nSaved merged result to {merged_file}")
    
    # Print final statistics
    print("\nClass statistics (after merging):")
    for class_name, examples in merged_classes.items():
        print(f"  {class_name}: {len(examples)} examples")
    
    return merged_classes

if __name__ == "__main__":
    input_file = "/mnt/public/sichuan_a/hyh/queryTest1/qaSchema/xtopDoc/V2/eval/extraction_results.json"
    output_dir = "/mnt/public/sichuan_a/hyh/queryTest1/qaSchema/xtopDoc/V2/eval/processed"
    
    merged_classes = process_and_merge(input_file, output_dir)