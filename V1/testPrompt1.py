import os
import sys
import json
import re
from tqdm import tqdm

print("="*50)
print("Schema实例化脚本")
print(f"Python版本: {sys.version}")
print("="*50)
sys.stdout.flush()

# 文件路径
prompt_path = "/mnt/public/sichuan_a/hyh/queryTest1/qaSchema/prompt.txt"
content_path = "/mnt/public/sichuan_a/hyh/queryTest1/qaSchema/xtopDoc/extracted_content.txt"
output_path = "/mnt/public/sichuan_a/hyh/queryTest1/qaSchema/xtopDoc/schema_output.jsonl"

# 检查文件
if not os.path.exists(prompt_path):
    print(f"错误: prompt文件不存在: {prompt_path}")
    sys.exit(1)

if not os.path.exists(content_path):
    print(f"错误: content文件不存在: {content_path}")
    sys.exit(1)

try:
    from openai import OpenAI
    print("OpenAI库导入成功")
except ImportError as e:
    print(f"错误: 无法导入OpenAI库: {e}")
    sys.exit(1)

# API配置
host = "http://120.25.194.125:2105/v1"
apikey = "xxx"
model = "qwen_docqa"

client = OpenAI(
    base_url=host,
    api_key=apikey,
    timeout=300.0
)
print("OpenAI客户端初始化成功")

# 读取prompt模板
with open(prompt_path, 'r', encoding='utf-8') as f:
    prompt_template = f.read()
print(f"prompt长度: {len(prompt_template)} 字符")

# 读取content文件
with open(content_path, 'r', encoding='utf-8') as f:
    content = f.read()
print(f"content长度: {len(content)} 字符")

# 解析chunks
chunks = []
current_chunk_id = None
current_content_lines = []

for line in content.split('\n'):
    if line.startswith('_') and not line.startswith('__'):
        if current_chunk_id is not None and current_content_lines:
            chunks.append({
                "chunk_id": current_chunk_id,
                "content": '\n'.join(current_content_lines).strip()
            })
        current_chunk_id = line.strip()
        current_content_lines = []
    else:
        current_content_lines.append(line)

if current_chunk_id is not None and current_content_lines:
    chunks.append({
        "chunk_id": current_chunk_id,
        "content": '\n'.join(current_content_lines).strip()
    })

print(f"共有 {len(chunks)} 个chunk")
sys.stdout.flush()

def remove_think_tags(text):
    """移除<think>...</think>标签及其内容"""
    # 移除<think>...</think>包裹的内容（包括换行）
    result = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    return result.strip()

def clean_and_parse_jsonl(raw_output):
    """清理并解析JSONL输出"""
    # 1. 移除思考过程
    result = remove_think_tags(raw_output)
    
    # 2. 移除markdown代码块标记
    result = result.strip()
    if result.startswith("```json"):
        result = result[7:]
    elif result.startswith("```"):
        result = result[3:]
    if result.endswith("```"):
        result = result[:-3]
    result = result.strip()
    
    # 3. 逐行解析JSON
    valid_objects = []
    for line in result.split('\n'):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if "__class" in obj:
                valid_objects.append(obj)
        except json.JSONDecodeError:
            continue
    
    return valid_objects

def process_chunk(doc_content):
    """调用API进行schema实例化"""
    full_prompt = prompt_template.replace("<<DOC>>", doc_content)
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": full_prompt}
        ],
        temperature=0.3,
        max_tokens=8192
    )
    
    return response.choices[0].message.content

print("开始调用API...")
sys.stdout.flush()

class_counts = {
    "Command": 0, "Argument": 0, "Parameter": 0, "Example": 0,
    "Mode": 0, "File": 0, "Fail Reasons": 0, "Issues": 0,
    "Concept": 0, "Operation": 0, "Coverage": 0
}

with open(output_path, 'w', encoding='utf-8') as out_f:
    for chunk in tqdm(chunks, desc="Schema实例化"):
        chunk_id = chunk["chunk_id"]
        chunk_content = chunk["content"]
        
        if not chunk_content.strip():
            tqdm.write(f"⚠ {chunk_id} 内容为空，跳过")
            continue
        
        try:
            raw_result = process_chunk(chunk_content)
            objects = clean_and_parse_jsonl(raw_result)
            
            for obj in objects:
                obj["__source_chunk"] = chunk_id
                out_f.write(json.dumps(obj, ensure_ascii=False) + '\n')
                
                cls = obj.get("__class", "Unknown")
                if cls in class_counts:
                    class_counts[cls] += 1
            
            tqdm.write(f"✓ {chunk_id}: 提取 {len(objects)} 条")
            
        except Exception as e:
            tqdm.write(f"✗ {chunk_id} 出错: {e}")

print(f"\n完成！结果已写入: {output_path}")
print("\n各类实例统计:")
for cls, count in class_counts.items():
    if count > 0:
        print(f"  {cls}: {count}")