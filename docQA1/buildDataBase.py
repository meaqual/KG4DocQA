# =====================bulidDataBase.py===================
import json
from pathlib import Path


def extract_kg_database(input_path, output_path=None):
    """
    从 KG 中提取每个实例的关键字段，构建简洁的数据库
    
    输出格式: {id: "usage | description | scenarios"}
    """
    
    # 默认输出路径
    if output_path is None:
        p = Path(input_path)
        output_path = p.parent / f"{p.stem}_database.json"
    
    # 读取 KG
    with open(input_path, 'r', encoding='utf-8') as f:
        kg = json.load(f)
    
    database = {}
    stats = {
        "total": 0,
        "with_usage": 0,
        "with_description": 0,
        "with_scenarios": 0,
        "empty": 0
    }
    
    # 遍历每个类别
    for class_name, items in kg.items():
        for item in items:
            item_id = item.get("id")
            if not item_id:
                print(f"⚠️ 跳过无 ID 的实例: {item.get('name', 'unknown')}")
                continue
            
            # 提取字段
            parts = []
            
            usage = item.get("usage", "").strip()
            description = item.get("description", "").strip()
            scenarios = item.get("scenarios", "")
            
            # 处理 scenarios（可能是列表或字符串）
            if isinstance(scenarios, list):
                scenarios = "; ".join(str(s).strip() for s in scenarios if s)
            elif isinstance(scenarios, str):
                scenarios = scenarios.strip()
            else:
                scenarios = ""
            
            # 组合非空字段
            if usage:
                parts.append(usage)
                stats["with_usage"] += 1
            if description:
                parts.append(description)
                stats["with_description"] += 1
            if scenarios:
                parts.append(f"应用场景: {scenarios}")
                stats["with_scenarios"] += 1
            
            # 构建值
            if parts:
                value = " | ".join(parts)
            else:
                value = ""
                stats["empty"] += 1
            
            database[item_id] = value
            stats["total"] += 1
    
    # 保存结果
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(database, f, ensure_ascii=False, indent=2)
    
    # 打印统计
    print(f"✅ 完成！已构建 KG 数据库")
    print(f"   输入: {input_path}")
    print(f"   输出: {output_path}")
    print(f"\n📊 统计:")
    print(f"   - 总条目: {stats['total']}")
    print(f"   - 有 usage: {stats['with_usage']}")
    print(f"   - 有 description: {stats['with_description']}")
    print(f"   - 有 scenarios: {stats['with_scenarios']}")
    print(f"   - 空值条目: {stats['empty']}")
    
    return database


def extract_kg_database_detailed(input_path, output_path=None):
    """
    提取为结构化格式（保留字段分离）
    
    输出格式: {id: {usage: "...", description: "...", scenarios: "..."}}
    """
    
    if output_path is None:
        p = Path(input_path)
        output_path = p.parent / f"{p.stem}_database_detailed.json"
    
    with open(input_path, 'r', encoding='utf-8') as f:
        kg = json.load(f)
    
    database = {}
    
    for class_name, items in kg.items():
        for item in items:
            item_id = item.get("id")
            if not item_id:
                continue
            
            entry = {}
            
            # 提取 usage
            if item.get("usage"):
                entry["usage"] = item["usage"].strip()
            
            # 提取 description
            if item.get("description"):
                entry["description"] = item["description"].strip()
            
            # 提取 scenarios
            scenarios = item.get("scenarios", "")
            if isinstance(scenarios, list):
                scenarios = "; ".join(str(s).strip() for s in scenarios if s)
            if scenarios:
                entry["scenarios"] = scenarios.strip()
            
            # 只保存非空条目
            if entry:
                database[item_id] = entry
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(database, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 完成！已构建详细数据库")
    print(f"   输出: {output_path}")
    print(f"   条目数: {len(database)}")
    
    return database


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python extract_kg_db.py <input.json> [--detailed]")
        print("示例: python extract_kg_db.py merged_classes_with_id.json")
        print("      python extract_kg_db.py merged_classes_with_id.json --detailed")
        sys.exit(1)
    
    input_path = sys.argv[1]
    detailed = "--detailed" in sys.argv
    
    if detailed:
        extract_kg_database_detailed(input_path)
    else:
        extract_kg_database(input_path)