import re
import json

def extract_json_from_schema_output(input_file, output_file):
    """
    从 schema_output.txt 中提取所有有效的 JSON 对象
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 移除 <think>...</think> 块
    content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
    
    # 移除批次标记行（如 "=== 第 1 批结果 ==="）
    content = re.sub(r'===.*?===', '', content)
    
    # 存储所有有效的 JSON 对象
    json_objects = []
    
    # 按行处理，尝试解析每一行作为 JSON
    lines = content.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # 尝试解析为 JSON
        try:
            # 处理可能的 JSON 对象
            if line.startswith('{') and line.endswith('}'):
                obj = json.loads(line)
                json_objects.append(obj)
            elif line.startswith('{'):
                # 可能是多行 JSON，暂时跳过单行处理
                # 对于格式良好的 JSONL，每行应该是完整的 JSON
                pass
        except json.JSONDecodeError:
            # 不是有效的 JSON，跳过
            continue
    
    # 写入输出文件（JSONL 格式）
    with open(output_file, 'w', encoding='utf-8') as f:
        for obj in json_objects:
            f.write(json.dumps(obj, ensure_ascii=False) + '\n')
    
    print(f"提取完成！共提取 {len(json_objects)} 个 JSON 对象")
    print(f"输出文件: {output_file}")
    
    # 统计各类实例数量
    class_counts = {}
    for obj in json_objects:
        cls = obj.get('__class', 'Unknown')
        class_counts[cls] = class_counts.get(cls, 0) + 1
    
    print("\n各类实例统计:")
    for cls, count in sorted(class_counts.items()):
        print(f"  {cls}: {count}")

if __name__ == "__main__":
    input_file = "/mnt/public/sichuan_a/hyh/queryTest1/qaSchema/xtopDoc/schema_output.txt"
    output_file = "/mnt/public/sichuan_a/hyh/queryTest1/qaSchema/xtopDoc/extracted_json.jsonl"
    extract_json_from_schema_output(input_file, output_file)