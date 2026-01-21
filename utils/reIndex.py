# =======================reIndex.py==========================
import json
import sys
from pathlib import Path


def reindex_kg(input_path, output_path=None, id_prefix="kg"):
    """
    重新为 KG 中的每个实例分配连续的 ID
    
    ID 格式: {prefix}_{class}_{序号}
    例如: kg_Command_0001, kg_Argument_0002, kg_Concept_0003
    """
    
    # 默认输出路径
    if output_path is None:
        p = Path(input_path)
        output_path = p.parent / f"{p.stem}_reindexed.json"
    
    # 读取 KG
    with open(input_path, 'r', encoding='utf-8') as f:
        kg = json.load(f)
    
    stats = {}
    total_count = 0
    
    # 遍历每个类别
    for class_name, items in kg.items():
        stats[class_name] = len(items)
        
        for idx, item in enumerate(items, 1):
            # 生成新的连续 ID
            new_id = f"{id_prefix}_{class_name}_{idx:04d}"
            old_id = item.get("id", "NO_ID")
            
            # 更新 ID（保持 id 在第一位）
            new_item = {"id": new_id}
            for key, value in item.items():
                if key != "id":
                    new_item[key] = value
            
            items[idx - 1] = new_item
            total_count += 1
    
    # 保存结果
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(kg, f, ensure_ascii=False, indent=2)
    
    # 打印统计信息
    print(f"\n{'='*60}")
    print(f"📋 重新索引报告")
    print(f"{'='*60}")
    print(f"   输入文件: {input_path}")
    print(f"   输出文件: {output_path}")
    
    print(f"\n📊 各类别统计:")
    for class_name, count in stats.items():
        print(f"   - {class_name}: {count} 条 (ID: {id_prefix}_{class_name}_0001 ~ {id_prefix}_{class_name}_{count:04d})")
    
    print(f"   ─────────────────────────────")
    print(f"   总计: {total_count} 条")
    print(f"\n✅ 重新索引完成！所有 ID 现在是连续的。")
    print(f"{'='*60}")
    
    return kg


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python reindex_kg.py <input.json> [output.json] [prefix]")
        print("示例: python reindex_kg.py textWithId.json")
        print("      python reindex_kg.py textWithId.json reindexed.json")
        print("      python reindex_kg.py textWithId.json reindexed.json eda")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    prefix = sys.argv[3] if len(sys.argv) > 3 else "kg"
    
    reindex_kg(input_path, output_path, prefix)