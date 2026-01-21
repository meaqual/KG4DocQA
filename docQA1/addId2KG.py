import json
from pathlib import Path


def add_ids_to_kg(input_path, output_path=None, id_prefix="kg"):
    """
    为 KG 中的每个实例添加顺序 ID
    
    ID 格式: {prefix}_{class}_{序号}
    例如: kg_Command_0001, kg_Argument_0042, kg_Concept_0103
    """
    
    # 默认输出路径
    if output_path is None:
        p = Path(input_path)
        output_path = p.parent / f"{p.stem}_with_id{p.suffix}"
    
    # 读取 KG
    with open(input_path, 'r', encoding='utf-8') as f:
        kg = json.load(f)
    
    stats = {}
    total_count = 0
    
    # 遍历每个类别
    for class_name, items in kg.items():
        stats[class_name] = len(items)
        
        for idx, item in enumerate(items, 1):
            # 生成顺序 ID
            item_id = f"{id_prefix}_{class_name}_{idx:04d}"
            
            # 直接在原字典中添加 id 字段
            # 如果想让 id 在最前面，需要重建字典
            if isinstance(item, dict):
                # 将 id 放在字典第一位
                new_item = {"id": item_id}
                new_item.update(item)
                items[idx - 1] = new_item
            else:
                # 如果 item 不是字典（比如是字符串），转换为字典格式
                items[idx - 1] = {
                    "id": item_id,
                    "content": item
                }
            
            total_count += 1
    
    # 保存结果
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(kg, f, ensure_ascii=False, indent=2)
    
    # 打印统计信息
    print(f"✅ 完成！已为 KG 添加 ID")
    print(f"   输入: {input_path}")
    print(f"   输出: {output_path}")
    print(f"\n📊 统计:")
    for class_name, count in stats.items():
        print(f"   - {class_name}: {count} 条")
    print(f"   ─────────────────")
    print(f"   总计: {total_count} 条")
    
    return kg


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python add_kg_id.py <input.json> [output.json] [prefix]")
        print("示例: python add_kg_id.py merged_classes.json")
        print("      python add_kg_id.py merged_classes.json output.json eda")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    prefix = sys.argv[3] if len(sys.argv) > 3 else "kg"
    
    add_ids_to_kg(input_path, output_path, prefix)