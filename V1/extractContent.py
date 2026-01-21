import json

# 读取gt_benchmark.json，提取前50个id的reference_doc_id
gt_benchmark_path = "/mnt/public/sichuan_a/hyh/queryTest1/qaSchema/xtopDoc/gt_benchmark.json"
knowledge_path = "/mnt/public/sichuan_a/hyh/queryTest1/qaSchema/xtopDoc/knowledge_xtop_chunked_768.json"
output_path = "/mnt/public/sichuan_a/hyh/queryTest1/qaSchema/xtopDoc/extracted_content.txt"

# 读取gt_benchmark.json
with open(gt_benchmark_path, 'r', encoding='utf-8') as f:
    gt_data = json.load(f)

# 提取前50个id的reference_doc_id
reference_doc_ids = []
for item in gt_data[:50]:
    if "reference_doc_id" in item:
        reference_doc_ids.extend(item["reference_doc_id"])

# 去重
reference_doc_ids = list(set(reference_doc_ids))
print(f"提取到的reference_doc_id数量（去重后）: {len(reference_doc_ids)}")

# 读取knowledge_xtop_chunked_768.json
with open(knowledge_path, 'r', encoding='utf-8') as f:
    knowledge_data = json.load(f)

# 构建chunk_id到content的映射
chunk_id_to_content = {}
for source_item in knowledge_data:
    if "knowledge" in source_item:
        for chunk in source_item["knowledge"]:
            chunk_id = chunk.get("chunk_id", "")
            content = chunk.get("content", "")
            chunk_id_to_content[chunk_id] = content

# 根据reference_doc_id找到对应的content并写入txt
with open(output_path, 'w', encoding='utf-8') as f:
    for doc_id in reference_doc_ids:
        if doc_id in chunk_id_to_content:
            f.write(f"{doc_id}\n")
            f.write(f"{chunk_id_to_content[doc_id]}\n\n")
        else:
            print(f"未找到chunk_id: {doc_id}")

print(f"结果已写入: {output_path}")