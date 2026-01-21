import json
import os
import time
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
    unnamed_counter = 0
    
    for class_name, examples in all_classes.items():
        for example in examples:
            name = example.get("name", None)
            if name is None or name == "":
                # If no name field, use a unique key
                key = (class_name, f"_unnamed_{unnamed_counter}")
                unnamed_counter += 1
            else:
                key = (class_name, str(name))
            
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(example)
    
    return grouped


def is_empty_value(value):
    """
    Check if a value is considered empty.
    """
    if value is None:
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    if isinstance(value, list) and len(value) == 0:
        return True
    if isinstance(value, dict) and len(value) == 0:
        return True
    # Check for list of empty dicts or all empty items
    if isinstance(value, list):
        if all(is_empty_value(item) for item in value):
            return True
    return False


def calculate_value_score(value):
    """
    计算一个值的"完整度"分数。
    """
    if value is None:
        return 0
    if isinstance(value, str):
        return len(value.strip())
    if isinstance(value, (int, float, bool)):
        return 1
    if isinstance(value, list):
        if len(value) == 0:
            return 0
        return sum(calculate_value_score(item) for item in value) + len(value)
    if isinstance(value, dict):
        if len(value) == 0:
            return 0
        return sum(calculate_value_score(v) for v in value.values()) + len(value)
    return 1


def get_best_value(values_list):
    """
    从多个值中选择最佳的非空值。
    如果全部为空，返回 None。
    """
    non_empty_values = [v for v in values_list if not is_empty_value(v)]
    
    if not non_empty_values:
        # 所有值都是空的，返回 None
        return None
    
    if len(non_empty_values) == 1:
        return non_empty_values[0]
    
    # 多个非空值，选择最完整的
    best_value = non_empty_values[0]
    best_score = calculate_value_score(best_value)
    
    for value in non_empty_values[1:]:
        score = calculate_value_score(value)
        if score > best_score:
            best_value = value
            best_score = score
    
    return best_value


def items_match(item1, item2):
    """
    Check if two dict items represent the same thing.
    """
    if not isinstance(item1, dict) or not isinstance(item2, dict):
        return False
    
    for key in ['name', 'type', 'value', 'id']:
        if key in item1 and key in item2:
            v1, v2 = item1[key], item2[key]
            if v1 is not None and v2 is not None and v1 == v2:
                return True
    
    if set(item1.keys()) == set(item2.keys()):
        return True
    
    return False


def merge_dict_lists(base_list, other_list):
    """
    Merge two lists of dicts.
    """
    if is_empty_value(other_list):
        return base_list if not is_empty_value(base_list) else None
    if is_empty_value(base_list):
        return other_list if not is_empty_value(other_list) else None
    
    result = []
    for base_item in base_list:
        merged_item = base_item.copy() if isinstance(base_item, dict) else base_item
        
        if isinstance(merged_item, dict):
            for other_item in other_list:
                if isinstance(other_item, dict) and items_match(base_item, other_item):
                    for key, value in other_item.items():
                        if key not in merged_item or is_empty_value(merged_item[key]):
                            if not is_empty_value(value):
                                merged_item[key] = value
                    break
        
        result.append(merged_item)
    
    return result if result else None


def merge_lists(list1, list2):
    """
    Merge two lists intelligently.
    """
    if is_empty_value(list1) and is_empty_value(list2):
        return None
    if is_empty_value(list1):
        return list2
    if is_empty_value(list2):
        return list1
    
    # Check if lists contain dicts
    if all(isinstance(item, dict) for item in list1) and all(isinstance(item, dict) for item in list2):
        score1 = calculate_value_score(list1)
        score2 = calculate_value_score(list2)
        
        if score1 >= score2:
            return merge_dict_lists(list1, list2)
        else:
            return merge_dict_lists(list2, list1)
    
    # For simple lists, combine and deduplicate
    combined = list(list1)
    for item in list2:
        if item not in combined:
            combined.append(item)
    return combined


def manual_merge(examples):
    """
    Manually merge examples when Qwen fails.
    Intelligently selects non-empty values and merges lists/dicts.
    If all values for a field are empty, keeps None.
    """
    if len(examples) == 0:
        return {}
    if len(examples) == 1:
        return examples[0]
    
    # 收集所有字段名
    all_keys = set()
    chunk_ids = []
    
    for example in examples:
        if isinstance(example, dict):
            all_keys.update(example.keys())
            # 收集 chunk_id
            cid = example.get("chunk_id")
            if cid:
                if isinstance(cid, list):
                    chunk_ids.extend(cid)
                else:
                    chunk_ids.append(cid)
    
    all_keys.discard("chunk_id")  # 单独处理 chunk_id
    
    merged = {}
    
    # 对每个字段，收集所有实例的值，选择最佳值
    for key in all_keys:
        values_for_key = []
        for example in examples:
            if isinstance(example, dict) and key in example:
                values_for_key.append(example[key])
        
        if not values_for_key:
            merged[key] = None
        else:
            best_value = get_best_value(values_for_key)
            merged[key] = best_value
    
    # 处理 chunk_ids
    if chunk_ids:
        unique_chunk_ids = list(dict.fromkeys(chunk_ids))
        merged["chunk_id"] = unique_chunk_ids if len(unique_chunk_ids) > 1 else unique_chunk_ids[0]
    
    return merged


def extract_json_from_text(text):
    """
    Try multiple methods to extract JSON from text.
    """
    if not text or not text.strip():
        return None
    
    text = text.strip()
    
    # Method 1: Direct parse
    try:
        return json.loads(text)
    except:
        pass
    
    # Method 2: Remove markdown code blocks
    if "```json" in text:
        try:
            json_text = text.split("```json")[1].split("```")[0].strip()
            return json.loads(json_text)
        except:
            pass
    
    if "```" in text:
        try:
            parts = text.split("```")
            for part in parts[1::2]:
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                try:
                    return json.loads(part)
                except:
                    continue
        except:
            pass
    
    # Method 3: Remove thinking tags
    if "<think>" in text:
        try:
            text_after_think = text.split("</think>")[-1].strip()
            result = json.loads(text_after_think)
            return result
        except:
            pass
    
    # Method 4: Find JSON object boundaries
    try:
        start_idx = text.find("{")
        if start_idx != -1:
            brace_count = 0
            for i, char in enumerate(text[start_idx:], start_idx):
                if char == "{":
                    brace_count += 1
                elif char == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        json_text = text[start_idx:i+1]
                        return json.loads(json_text)
    except:
        pass
    
    # Method 5: Find JSON array boundaries
    try:
        start_idx = text.find("[")
        if start_idx != -1:
            bracket_count = 0
            for i, char in enumerate(text[start_idx:], start_idx):
                if char == "[":
                    bracket_count += 1
                elif char == "]":
                    bracket_count -= 1
                    if bracket_count == 0:
                        json_text = text[start_idx:i+1]
                        return json.loads(json_text)
    except:
        pass
    
    return None


def merge_examples_with_qwen(examples, class_name, max_retries=3):
    """
    Use Qwen to merge multiple examples with the same class and name.
    """
    if len(examples) == 1:
        return examples[0]
    
    # Check if examples are too large
    try:
        examples_json = json.dumps(examples, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"    Error serializing examples: {e}")
        return manual_merge(examples)
    
    if len(examples_json) > 12000:
        print(f"    Warning: Examples too large ({len(examples_json)} chars), using manual merge")
        return manual_merge(examples)
    
    prompt = f"""你是一个EDA文档知识合并专家。以下是多个关于同一个{class_name}的提取结果，它们具有相同的class和name。
请将这些结果合并成一个完整的条目，要求：
1. 保持原有的字段结构不变
2. 合并时选择最完整、最详细的信息
3. 如果某个字段在不同条目中有不同的值，选择最详细或最准确的那个
4. chunk_id字段应该合并为一个列表，包含所有来源的chunk_id
5. 如果有冲突的信息，优先保留更详细的描述
6. 如果某个字段在所有条目中都为空(null、[]、"")，则保持为null

需要合并的条目：
{examples_json}

请只返回合并后的JSON对象，格式如下：
{{
  "class": "...",
  "name": "...",
  ...其他字段...
  "chunk_id": ["id1", "id2", ...]
}}

不要包含任何解释或其他文字，只返回JSON。"""

    for retry in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system", 
                        "content": "你是一个JSON合并专家。你只返回有效的JSON对象，不包含任何其他文字、解释或markdown标记。"
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=4096,
            )
            
            if not response or not response.choices:
                print(f"    Retry {retry+1}/{max_retries}: Empty response object")
                time.sleep(1)
                continue
            
            result_text = response.choices[0].message.content
            
            if not result_text:
                print(f"    Retry {retry+1}/{max_retries}: Empty content in response")
                time.sleep(1)
                continue
            
            merged = extract_json_from_text(result_text)
            
            if merged is not None:
                return merged
            else:
                print(f"    Retry {retry+1}/{max_retries}: Could not extract JSON from response")
                print(f"    Response preview: {result_text[:200]}...")
                time.sleep(1)
                continue
            
        except json.JSONDecodeError as e:
            print(f"    Retry {retry+1}/{max_retries}: JSON decode error: {e}")
            time.sleep(1)
            continue
        except Exception as e:
            print(f"    Retry {retry+1}/{max_retries}: API error: {type(e).__name__}: {e}")
            time.sleep(2)
            continue
    
    print(f"    All retries failed for {class_name}, using manual merge")
    return manual_merge(examples)


def process_and_merge(input_file, output_dir):
    """
    Main function to process the extraction results and merge duplicates.
    """
    # Load data
    print("=" * 60)
    print("Loading extraction results...")
    data = load_extraction_results(input_file)
    print(f"Loaded {len(data)} chunks")
    
    # Extract classes with chunk_id
    print("\n" + "=" * 60)
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
    total_before = 0
    for class_name, examples in sorted(all_classes.items()):
        print(f"  {class_name}: {len(examples)} examples")
        total_before += len(examples)
    print(f"  TOTAL: {total_before} examples")
    
    # Group by (class, name)
    print("\n" + "=" * 60)
    print("Grouping by (class, name)...")
    grouped = group_by_class_and_name(all_classes)
    
    # Find duplicates (groups with more than 1 example)
    duplicates = {k: v for k, v in grouped.items() if len(v) > 1}
    singletons = {k: v for k, v in grouped.items() if len(v) == 1}
    
    print(f"Total groups: {len(grouped)}")
    print(f"  - Groups with duplicates (need merging): {len(duplicates)}")
    print(f"  - Single item groups (no merge needed): {len(singletons)}")
    
    # Show duplicate statistics by class
    if duplicates:
        print("\nDuplicate groups by class:")
        dup_by_class = {}
        for (class_name, name), examples in duplicates.items():
            if class_name not in dup_by_class:
                dup_by_class[class_name] = 0
            dup_by_class[class_name] += 1
        for class_name, count in sorted(dup_by_class.items()):
            print(f"  {class_name}: {count} groups need merging")
    
    # Merge duplicates using Qwen
    print("\n" + "=" * 60)
    print("Merging duplicates...")
    merged_classes = {}
    processed = 0
    total_groups = len(grouped)
    merge_count = 0
    
    for (class_name, name), examples in grouped.items():
        if class_name not in merged_classes:
            merged_classes[class_name] = []
        
        processed += 1
        
        if len(examples) > 1:
            display_name = name[:40] + "..." if len(name) > 40 else name
            print(f"  [{processed}/{total_groups}] Merging {len(examples)} examples: {class_name}/{display_name}")
            merged = merge_examples_with_qwen(examples, class_name)
            merge_count += 1
        else:
            merged = examples[0]
        
        merged_classes[class_name].append(merged)
        
        # Progress update every 50 items
        if processed % 50 == 0:
            print(f"  === Progress: {processed}/{total_groups} ({processed*100//total_groups}%) ===")
    
    # Save merged result
    print("\n" + "=" * 60)
    merged_file = os.path.join(output_dir, "merged_classes.json")
    with open(merged_file, 'w', encoding='utf-8') as f:
        json.dump(merged_classes, f, ensure_ascii=False, indent=2)
    print(f"Saved merged result to {merged_file}")
    
    # Print final statistics
    print("\nClass statistics (after merging):")
    total_after = 0
    for class_name, examples in sorted(merged_classes.items()):
        print(f"  {class_name}: {len(examples)} examples")
        total_after += len(examples)
    print(f"  TOTAL: {total_after} examples")
    
    # Summary
    if total_before > 0:
        reduction = total_before - total_after
        print(f"\n" + "=" * 60)
        print("Summary:")
        print(f"  Before merging: {total_before} examples")
        print(f"  After merging: {total_after} examples")
        print(f"  Reduced: {reduction} examples ({reduction * 100 / total_before:.1f}%)")
        print(f"  Merge operations: {merge_count}")
    
    return merged_classes


if __name__ == "__main__":
    input_file = "/mnt/public/sichuan_a/hyh/queryTest1/qaSchema/xtopDoc/V2/eval/extraction_results.json"
    output_dir = "/mnt/public/sichuan_a/hyh/queryTest1/qaSchema/xtopDoc/V2/eval/processed2"
    
    print("Starting extraction and merge process...")
    print(f"Input: {input_file}")
    print(f"Output: {output_dir}")
    print()
    
    merged_classes = process_and_merge(input_file, output_dir)
    
    if merged_classes:
        print("\n" + "=" * 60)
        print("Process completed successfully!")
    else:
        print("\n" + "=" * 60)
        print("Process failed!")