import json
import sys
from pathlib import Path


# 定义每个 class 的必需字段和可选字段
CLASS_SCHEMAS = {
    "Command": {
        "required": ["class", "name", "usage", "syntax", "scenarios"],
        "optional": ["arguments", "values"]
    },
    "Argument": {
        "required": ["class", "name", "command", "usage", "syntax", "scenarios"],
        "optional": ["values"]
    },
    "Parameter": {
        "required": ["class", "name", "usage", "type", "scenarios"],
        "optional": ["range", "key_values"]
    },
    "Example": {
        "required": ["class", "name", "usage", "scenarios"],
        "optional": []
    },
    "Mode": {
        "required": ["class", "name", "usage", "scenarios"],
        "optional": []
    },
    "File": {
        "required": ["class", "name", "usage"],
        "optional": []
    },
    "FailReasons": {
        "required": ["class", "name", "reasons", "description", "solution"],
        "optional": []
    },
    "Issues": {
        "required": ["class", "name", "descriptions"],
        "optional": []
    },
    "Task": {
        "required": ["class", "name", "description"],
        "optional": []
    },
    "Concept": {
        "required": ["class", "name", "description"],
        "optional": []
    },
    "Operation": {
        "required": ["class", "name", "description", "related_commands", "effect"],
        "optional": []
    }
}


def check_missing_fields(input_path, output_path=None):
    """
    检查 KG 中每个实例是否缺失必需字段
    """
    
    # 默认输出路径
    if output_path is None:
        p = Path(input_path)
        output_path = p.parent / f"{p.stem}_missing_fields.json"
    
    # 读取 KG
    with open(input_path, 'r', encoding='utf-8') as f:
        kg = json.load(f)
    
    missing_report = {}
    total_missing = 0
    total_checked = 0
    
    # 遍历每个类别
    for class_name, items in kg.items():
        if class_name not in CLASS_SCHEMAS:
            print(f"⚠️  未知类别: {class_name}，跳过检查")
            continue
        
        schema = CLASS_SCHEMAS[class_name]
        required_fields = schema["required"]
        class_missing = []
        
        for item in items:
            total_checked += 1
            item_id = item.get("id", "NO_ID")
            item_name = item.get("name", "NO_NAME")
            
            # 检查缺失的必需字段
            missing_fields = []
            for field in required_fields:
                if field not in item or item[field] is None or item[field] == "":
                    missing_fields.append(field)
            
            if missing_fields:
                total_missing += 1
                class_missing.append({
                    "id": item_id,
                    "name": item_name,
                    "missing_fields": missing_fields
                })
        
        if class_missing:
            missing_report[class_name] = class_missing
    
    # 保存报告
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(missing_report, f, ensure_ascii=False, indent=2)
    
    # 打印统计信息
    print(f"\n{'='*60}")
    print(f"📋 字段缺失检查报告")
    print(f"{'='*60}")
    print(f"   输入文件: {input_path}")
    print(f"   报告输出: {output_path}")
    print(f"\n📊 统计:")
    print(f"   检查总数: {total_checked} 条")
    print(f"   缺失总数: {total_missing} 条")
    print(f"   完整率: {((total_checked - total_missing) / total_checked * 100):.2f}%")
    
    if missing_report:
        print(f"\n❌ 各类别缺失详情:")
        for class_name, items in missing_report.items():
            print(f"\n   【{class_name}】 缺失 {len(items)} 条:")
            for item in items[:5]:  # 只显示前5条
                print(f"      - {item['id']} ({item['name']})")
                print(f"        缺失字段: {', '.join(item['missing_fields'])}")
            if len(items) > 5:
                print(f"      ... 还有 {len(items) - 5} 条，详见输出文件")
    else:
        print(f"\n✅ 所有实例的必需字段都完整！")
    
    print(f"\n{'='*60}")
    
    return missing_report


def print_schema_summary():
    """打印所有类别的字段要求"""
    print("\n📖 各类别必需字段一览:")
    print("-" * 40)
    for class_name, schema in CLASS_SCHEMAS.items():
        print(f"   {class_name}:")
        print(f"      必需: {', '.join(schema['required'])}")
        if schema['optional']:
            print(f"      可选: {', '.join(schema['optional'])}")
    print("-" * 40)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python check_missing_fields.py <input.json> [output.json]")
        print("示例: python check_missing_fields.py textWithId.json")
        print("      python check_missing_fields.py textWithId.json missing_report.json")
        print_schema_summary()
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    check_missing_fields(input_path, output_path)