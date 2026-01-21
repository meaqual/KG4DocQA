# ====================countOverlap.py============================
import json
import sys
from pathlib import Path
from collections import defaultdict


def find_duplicates(input_path, output_path=None):
    """
    找出 class 和 name 都重复的实例
    """
    
    # 默认输出路径
    if output_path is None:
        p = Path(input_path)
        output_path = p.parent / f"{p.stem}_duplicates.json"
    
    # 读取 KG
    with open(input_path, 'r', encoding='utf-8') as f:
        kg = json.load(f)
    
    # 用于存储 (class, name) -> [实例列表]
    class_name_map = defaultdict(list)
    
    # 遍历每个类别
    for class_name, items in kg.items():
        for item in items:
            item_class = item.get("class", class_name)
            item_name = item.get("name", "")
            key = (item_class, item_name)
            
            class_name_map[key].append({
                "id": item.get("id", "NO_ID"),
                "class": item_class,
                "name": item_name,
                "full_item": item
            })
    
    # 筛选出重复的
    duplicates = {}
    total_duplicate_groups = 0
    total_duplicate_items = 0
    
    for (item_class, item_name), instances in class_name_map.items():
        if len(instances) > 1:
            total_duplicate_groups += 1
            total_duplicate_items += len(instances)
            
            if item_class not in duplicates:
                duplicates[item_class] = []
            
            duplicates[item_class].append({
                "name": item_name,
                "count": len(instances),
                "ids": [inst["id"] for inst in instances],
                "instances": [inst["full_item"] for inst in instances]
            })
    
    # 保存报告
    # 简化版报告（不含完整实例）
    simple_report = {}
    for class_name, items in duplicates.items():
        simple_report[class_name] = [
            {
                "name": item["name"],
                "count": item["count"],
                "ids": item["ids"]
            }
            for item in items
        ]
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(simple_report, f, ensure_ascii=False, indent=2)
    
    # 详细报告（含完整实例）
    detailed_output_path = Path(output_path).parent / f"{Path(output_path).stem}_detailed.json"
    with open(detailed_output_path, 'w', encoding='utf-8') as f:
        json.dump(duplicates, f, ensure_ascii=False, indent=2)
    
    # 打印统计信息
    print(f"\n{'='*60}")
    print(f"📋 重复实例检查报告")
    print(f"{'='*60}")
    print(f"   输入文件: {input_path}")
    print(f"   简要报告: {output_path}")
    print(f"   详细报告: {detailed_output_path}")
    
    print(f"\n📊 统计:")
    print(f"   重复组数: {total_duplicate_groups} 组")
    print(f"   涉及实例: {total_duplicate_items} 条")
    
    if duplicates:
        print(f"\n❌ 各类别重复详情:")
        for class_name, items in duplicates.items():
            print(f"\n   【{class_name}】 {len(items)} 组重复:")
            for item in items[:10]:  # 只显示前10组
                print(f"      - \"{item['name']}\" 重复 {item['count']} 次")
                print(f"        IDs: {', '.join(item['ids'])}")
            if len(items) > 10:
                print(f"      ... 还有 {len(items) - 10} 组，详见输出文件")
    else:
        print(f"\n✅ 没有发现重复实例！")
    
    print(f"\n{'='*60}")
    
    return duplicates


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python find_duplicates.py <input.json> [output.json]")
        print("示例: python find_duplicates.py textWithId.json")
        print("      python find_duplicates.py textWithId.json duplicates.json")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    find_duplicates(input_path, output_path)