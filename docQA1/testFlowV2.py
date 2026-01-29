import argparse
import json
import re
import sys
from datetime import datetime
from openai import OpenAI
from tqdm import tqdm
from prompt import (
    STAGE1_IDENTIFY_CLASSES, STAGE1_GLEAN, STAGE2_PROMPTS,
    verify_schemas, write_verification_report
)

# ============ 全局配置 ============
HOST = "http://120.25.194.125:2105/v1"
APIKEY = "xxx"
MODEL = "qwen_docqa"
MAX_GLEAN_ROUNDS = 3

INPUT_FILE = "/mnt/public/sichuan_a/hyh/queryTest1/qaSchema/xtopDoc/V2/knowledge_xtop_chunked_768.json"
OUTPUT_TXT = "/mnt/public/sichuan_a/hyh/queryTest1/qaSchema/xtopDoc/V2/results/extraction_output.txt"
OUTPUT_JSON = "/mnt/public/sichuan_a/hyh/queryTest1/qaSchema/xtopDoc/V2/results/extraction_results.json"

# 默认处理范围 (None=全部, "5"=第5个, "5-10"=第5到10个)
DEFAULT_RANGE = "1"
# ================================

client = OpenAI(
    base_url=HOST,
    api_key=APIKEY,
    timeout=300.0
)
   

def parse_json_response(content):
    """尝试从响应中解析JSON，支持markdown格式"""
    if not content:
        return None
    
    content = content.strip()
    
    
    # 尝试提取markdown代码块中的JSON
    json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', content)
    if json_match:
        content = json_match.group(1).strip()
    
    # 尝试找到JSON对象的边界
    start_idx = content.find('{')
    if start_idx != -1:
        end_idx = content.rfind('}')
        if end_idx != -1:
            content = content[start_idx:end_idx+1]
    
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return None


def call_llm(prompt):
    """Call LLM and return JSON response"""
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    print(resp)
    return json.loads(resp.choices[0].message.content)


def stage1_identify(document, notes=""):
    """Stage 1: Identify instances with glean"""
    prompt = STAGE1_IDENTIFY_CLASSES.format(document=document, document_notes=notes)
    result = call_llm(prompt)
    all_instances = result.get("identified_classes", {})

    for _ in range(MAX_GLEAN_ROUNDS):
        prompt = STAGE1_GLEAN.format(
            previous_results=json.dumps(all_instances, ensure_ascii=False),
            document=document
        )
        glean = call_llm(prompt).get("additional_instances", {})

        added = False
        for cls, instances in glean.items():
            if instances:
                all_instances.setdefault(cls, []).extend(instances)
                added = True
        if not added:
            break

    return all_instances


def stage2_extract(document, identified):
    """Stage 2: Extract details for each instance"""
    extracted = {cls: [] for cls in identified}

    for cls, instances in identified.items():
        if cls not in STAGE2_PROMPTS:
            continue
        for instance in instances:
            prompt = STAGE2_PROMPTS[cls].format(instance=instance, document=document)
            data = call_llm(prompt)
            extracted[cls].append(data)

    return extracted
            
            
def stage3_verify(extracted):
    """Stage 3: Verify and return report string"""
    return verify_schemas(
        commands=extracted.get("Command", []),
        arguments=extracted.get("Argument", []),
        parameters=extracted.get("Parameter", []),
        operations=extracted.get("Operation", [])
    )


def load_knowledge_file(file_path):
    """加载知识库JSON文件，提取所有content和对应的chunk_id"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    chunks = []
    for source_item in data:
        source_name = source_item.get("source", "unknown")
        knowledge_list = source_item.get("knowledge", [])
        for knowledge_item in knowledge_list:
            chunk_id = knowledge_item.get("chunk_id", "unknown")
            doc_id = knowledge_item.get("doc_id", "unknown")
            content = knowledge_item.get("content", "")
            if content.strip():
                chunks.append({
                    "source": source_name,
                    "doc_id": doc_id,
                    "chunk_id": chunk_id,
                    "content": content
                })
    
    return chunks


def process_single_chunk(chunk_info, file_handle):
    """处理单个chunk，返回Stage2提取结果，同时写入文件"""
    chunk_id = chunk_info["chunk_id"]
    doc_id = chunk_info["doc_id"]
    source = chunk_info["source"]
    content = chunk_info["content"]
    
    output_lines = []
    output_lines.append(f"\n{'='*80}")
    output_lines.append(f"Processing: {chunk_id}")
    output_lines.append(f"Source: {source}")
    output_lines.append(f"Doc ID: {doc_id}")
    output_lines.append(f"{'='*80}")
    
    output_lines.append(f"\n--- Document Content ---")
    output_lines.append(content[:500] + "..." if len(content) > 500 else content)
    
    output_lines.append(f"\n=== Stage 1: Identify ===")
    identified = stage1_identify(content)
    output_lines.append(json.dumps(identified, indent=2, ensure_ascii=False))
    
    output_lines.append(f"\n=== Stage 2: Extract ===")
    extracted = stage2_extract(content, identified)
    output_lines.append(json.dumps(extracted, indent=2, ensure_ascii=False))
    
    output_lines.append(f"\n=== Stage 3: Verify ===")
    report = stage3_verify(extracted)
    output_lines.append(report)
    
    output_lines.append(f"\n{'='*80}")
    output_lines.append(f"==== END OF {chunk_id} ====")
    output_lines.append(f"{'='*80}\n")
    
    file_handle.write('\n'.join(output_lines) + '\n')
    file_handle.flush()
    
    return {
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "source": source,
        "extracted": extracted
    }


def parse_range(range_str, total_count):
    """
    解析范围字符串，返回 (start, end) 索引（0-based, end exclusive）
    
    支持格式:
      "5"     -> 第5个 (索引4-5)
      "5-10"  -> 第5到10个 (索引4-10)
      "5:"    -> 第5个到末尾
      ":10"   -> 第1到10个
      "-5"    -> 最后5个
    """
    range_str = range_str.strip()
    
    # 最后n个: "-5"
    if range_str.startswith('-') and ':' not in range_str:
        try:
            n = int(range_str)
            return max(0, total_count + n), total_count
        except ValueError:
            pass
    
    # 切片语法: "5:", ":10", "5:10"
    if ':' in range_str:
        parts = range_str.split(':')
        start = int(parts[0]) - 1 if parts[0] else 0
        end = int(parts[1]) if parts[1] else total_count
        return max(0, start), min(total_count, end)
    
    # 范围语法: "5-10"
    if '-' in range_str:
        parts = range_str.split('-')
        start = int(parts[0]) - 1
        end = int(parts[1])
        return max(0, start), min(total_count, end)
    
    # 单个数字: "5"
    n = int(range_str)
    return n - 1, n


def main():
    parser = argparse.ArgumentParser(description='Schema Extraction from EDA documents')
    parser.add_argument('-r', '--range', type=str, default=DEFAULT_RANGE,
                        help='Chunk range: "5" (5th), "5-10" (5th to 10th), '
                             '"5:" (5th to end), ":10" (1st to 10th), "-5" (last 5)')
    parser.add_argument('-i', '--input', type=str, default=INPUT_FILE,
                        help='Input JSON file path')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Output base name (without extension)')
    
    args = parser.parse_args()
    
    input_file = args.input
    if args.output:
        output_txt = args.output + '.txt'
        output_json = args.output + '.json'
    else:
        output_txt = OUTPUT_TXT
        output_json = OUTPUT_JSON
    
    print(f"Schema Extraction Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Input file: {input_file}")
    print(f"Output TXT: {output_txt}")
    print(f"Output JSON: {output_json}")
    
    print(f"\nLoading knowledge file...")
    all_chunks = load_knowledge_file(input_file)
    total_count = len(all_chunks)
    print(f"Found {total_count} chunks in total")
    
    if args.range:
        start, end = parse_range(args.range, total_count)
        chunks = all_chunks[start:end]
        print(f"Processing range: {start+1} to {end} ({len(chunks)} chunks)")
    else:
        chunks = all_chunks
        print(f"Processing all {len(chunks)} chunks")
    
    if not chunks:
        print("No chunks to process!")
        return
    
    with open(output_txt, 'w', encoding='utf-8') as txt_file:
        txt_file.write(f"Schema Extraction Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        txt_file.write(f"Input file: {input_file}\n")
        txt_file.write(f"Total chunks in file: {total_count}\n")
        if args.range:
            txt_file.write(f"Processing range: {args.range} ({len(chunks)} chunks)\n")
        txt_file.write(f"{'#'*80}\n")
        
        all_stage2_results = []
        
        for chunk_info in tqdm(chunks, desc="Processing chunks", unit="chunk", 
                                ncols=100, dynamic_ncols=True):
            try:
                stage2_result = process_single_chunk(chunk_info, txt_file)
                all_stage2_results.append(stage2_result)
            except Exception as e:
                tqdm.write(f"Error processing {chunk_info['chunk_id']}: {e}")
                error_result = {
                    "chunk_id": chunk_info["chunk_id"],
                    "doc_id": chunk_info["doc_id"],
                    "source": chunk_info["source"],
                    "extracted": {},
                    "error": str(e)
                }
                all_stage2_results.append(error_result)
                
                txt_file.write(f"\n{'='*80}\n")
                txt_file.write(f"ERROR processing {chunk_info['chunk_id']}: {e}\n")
                txt_file.write(f"{'='*80}\n")
                txt_file.flush()
        
        txt_file.write(f"\n{'#'*80}\n")
        txt_file.write(f"Schema Extraction Completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        txt_file.write(f"Total chunks processed: {len(all_stage2_results)}\n")
    
    print(f"\nSaving Stage 2 results to {output_json}...")
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(all_stage2_results, f, indent=2, ensure_ascii=False)
    
    print(f"\nSchema Extraction Completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total chunks processed: {len(all_stage2_results)}")
    print(f"\nResults saved to:")
    print(f"  - TXT: {output_txt}")
    print(f"  - JSON: {output_json}")


if __name__ == "__main__":
    main()