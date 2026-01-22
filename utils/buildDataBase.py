# =====================bulidDataBase.py===================
import json
from pathlib import Path


# 需要提取的字段
EXTRACT_FIELDS = ["usage", "description", "scenarios"]


def extract_kg_database(input_path, output_path=None):
    """
    从 KG 中提取 usage / description / scenarios 字段，构建查询数据库
    
    输出格式: {text: id}
    一个实例的多个字段会拆成多条记录
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
        "total_instances": 0,
        "total_records": 0,
        "by_field": {"usage": 0, "description": 0, "scenarios": 0}
    }
    
    # 遍历每个类别
    for class_name, items in kg.items():
        for item in items:
            item_id = item.get("id")
            if not item_id:
                print(f"⚠️ 跳过无 ID 的实例: {item.get('name', 'unknown')}")
                continue
            
            stats["total_instances"] += 1
            
            # 提取每个字段
            for field in EXTRACT_FIELDS:
                value = item.get(field)
                if value is None:
                    continue
                
                # 处理值
                text = process_field_value(value)
                
                if text:
                    # 文本已存在则追加 id
                    if text in database:
                        existing = database[text]
                        if item_id not in existing.split(", "):
                            database[text] = f"{existing}, {item_id}"
                    else:
                        database[text] = item_id
                    
                    stats["total_records"] += 1
                    stats["by_field"][field] += 1
    
    # 保存结果
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(database, f, ensure_ascii=False, indent=2)
    
    # 打印统计
    print(f"\n{'='*60}")
    print(f"✅ 完成！已构建 KG 数据库")
    print(f"{'='*60}")
    print(f"   输入: {input_path}")
    print(f"   输出: {output_path}")
    print(f"\n📊 统计:")
    print(f"   - 实例总数: {stats['total_instances']}")
    print(f"   - 记录总数: {stats['total_records']}")
    print(f"   - 唯一文本数: {len(database)}")
    print(f"\n📝 各字段记录数:")
    for field, count in stats["by_field"].items():
        print(f"   - {field}: {count}")
    print(f"{'='*60}")
    
    return database


def process_field_value(value):
    """处理字段值，转换为字符串"""
    if value is None:
        return ""
    
    if isinstance(value, str):
        return value.strip()
    
    if isinstance(value, list):
        parts = [str(item).strip() for item in value if item]
        return "; ".join(parts)
    
    return str(value).strip()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python buildDatabase.py <input.json> [output.json]")
        print("示例: python buildDatabase.py merged_classes_with_id.json")
        print(f"\n提取字段: {', '.join(EXTRACT_FIELDS)}")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    extract_kg_database(input_path, output_path)