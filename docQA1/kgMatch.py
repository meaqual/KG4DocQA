# =========================kgMatch.py=============================

"""
KG 实例名匹配检索器 - 基于 name 字段匹配

遍历数据库中的 name，检查是否出现在 query 中
"""

import os
import re
import json
from pathlib import Path
from typing import List, Dict, Tuple, Set
from collections import defaultdict

os.environ['CUDA_VISIBLE_DEVICES'] = '5'

# ============ 路径配置 ============
DATABASE_PATH = "/mnt/public/sichuan_a/hyh/queryTest1/qaSchema/xtopDoc/docQA1/dataBase/textWithId.json"
BENCHMARK_PATH = "/mnt/public/sichuan_a/hyh/queryTest1/qaSchema/xtopDoc/docQA1/testData/gt_benchmark.json"
OUTPUT_PATH = "/mnt/public/sichuan_a/hyh/queryTest1/qaSchema/xtopDoc/docQA1/results/kgMatch_results.txt"


# ============ 配置 ============
MATCHER_CONFIG = {
    "CASE_SENSITIVE": False,      # 是否区分大小写
    "MIN_NAME_LENGTH": 2,         # 最小 name 长度
    "MAX_RESULTS": 50,            # 最大返回结果数
    "TRY_VARIANTS": True,         # 尝试变体匹配 (name中的_-空格互换)
    # 屏蔽词（全部用小写）
    "BLOCKED_NAMES": {
        "xtop", "xtopio", "xtop_io",
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "of", "to", "in", "on", "for", "with", "at", "by", "from",
        "and", "or", "but", "if", "then", "else",
        "it", "this", "that", "these", "those",
        "do", "does", "did", "can", "could", "will", "would",
        "how", "what", "why", "when", "where", "which", "who",
    },
}


class KGNameMatcher:
    """
    KG 实例名匹配器
    
    核心逻辑：
    1. 加载数据库所有 name，按长度降序排列
    2. 遍历每个 name，检查是否出现在 query 中
    3. 匹配成功后标记 query 中对应位置，避免短 name 重复匹配
    4. 支持变体匹配：name 中的 _/- 可以匹配 query 中的空格
    """
    
    def __init__(
        self,
        kg_file_path: str = None,
        verbose: bool = True,
    ):
        self.verbose = verbose
        self.kg_file_path = kg_file_path
        
        # 所有实例列表
        self.instances: List[Dict] = []
        # name -> instances 的映射
        self.name_to_instances: Dict[str, List[Dict]] = defaultdict(list)
        # 所有 name（按长度降序排列）
        self.sorted_names: List[str] = []
        # name -> 预编译的匹配模式列表
        self.name_patterns: Dict[str, List[Tuple[re.Pattern, str]]] = {}
        
        self._load_and_build_index()
        
        if self.verbose:
            print(f"KG 实例名匹配器初始化完成")
            print(f"   - 实例数量: {len(self.instances)}")
            print(f"   - 唯一 name 数量: {len(self.name_to_instances)}")
    
    def _normalize_key(self, name: str) -> str:
        """标准化 name 作为索引 key"""
        if MATCHER_CONFIG["CASE_SENSITIVE"]:
            return name
        return name.lower()
    
    def _is_blocked(self, name: str) -> bool:
        """检查是否为屏蔽词"""
        return name.lower() in MATCHER_CONFIG["BLOCKED_NAMES"]
    
    def _generate_match_variants(self, name: str) -> List[str]:
        """
        生成 name 的匹配变体
        
        例如 "size_down" -> ["size_down", "size-down", "size down"]
        例如 "clock gating" -> ["clock gating", "clock_gating", "clock-gating"]
        """
        variants = [name]
        
        if not MATCHER_CONFIG["TRY_VARIANTS"]:
            return variants
        
        # 检查是否包含分隔符
        has_underscore = '_' in name
        has_hyphen = '-' in name
        has_space = ' ' in name
        
        if has_underscore or has_hyphen or has_space:
            # 统一替换为各种分隔符版本
            # 先统一成空格，再生成其他版本
            normalized = name.replace('_', ' ').replace('-', ' ')
            
            variants = [
                name,                              # 原始
                normalized,                        # 空格版
                normalized.replace(' ', '_'),     # 下划线版
                normalized.replace(' ', '-'),     # 连字符版
            ]
            # 去重
            variants = list(dict.fromkeys(variants))
        
        return variants
    
    def _build_pattern(self, text: str) -> re.Pattern:
        """
        构建匹配模式
        
        使用词边界避免子串误匹配：
        - "setup" 不应匹配 "setup" 中的 "top"
        - 但 "set_timing_derate" 应该能完整匹配
        """
        escaped = re.escape(text)
        # 词边界：前后不能是字母/数字/下划线
        pattern = rf'(?<![a-zA-Z0-9_]){escaped}(?![a-zA-Z0-9_])'
        flags = re.IGNORECASE if not MATCHER_CONFIG["CASE_SENSITIVE"] else 0
        return re.compile(pattern, flags)
    
    def _load_and_build_index(self):
        """加载数据并构建索引"""
        if self.verbose:
            print(f"加载 KG 文件: {self.kg_file_path}")
        
        with open(self.kg_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 遍历所有类别
        for category, items in data.items():
            if not isinstance(items, list):
                continue
            
            for item in items:
                if not isinstance(item, dict):
                    continue
                
                instance_id = item.get("id", "")
                name = item.get("name", "")
                description = item.get("description", "")
                
                if not name or len(name) < MATCHER_CONFIG["MIN_NAME_LENGTH"]:
                    continue
                
                # 过滤屏蔽词
                if self._is_blocked(name):
                    continue
                
                instance = {
                    "id": instance_id,
                    "name": name,
                    "class": item.get("class", category),
                    "description": description,
                    "chunk_id": item.get("chunk_id", ""),
                    "raw": item,
                }
                
                self.instances.append(instance)
                
                # 索引：name_key -> instances
                name_key = self._normalize_key(name)
                self.name_to_instances[name_key].append(instance)
        
        # 按 name 长度降序排列（优先匹配长的）
        self.sorted_names = sorted(
            self.name_to_instances.keys(),
            key=len,
            reverse=True
        )
        
        # 预编译每个 name 的匹配模式（包括变体）
        if self.verbose:
            print(f"   - 预编译匹配模式...")
        
        for name_key in self.sorted_names:
            # 获取原始 name（取第一个实例的 name）
            original_name = self.name_to_instances[name_key][0]["name"]
            
            # 生成变体
            variants = self._generate_match_variants(original_name)
            
            # 为每个变体编译模式
            patterns = []
            for variant in variants:
                pattern = self._build_pattern(variant)
                patterns.append((pattern, variant))
            
            self.name_patterns[name_key] = patterns
        
        if self.verbose:
            print(f"   - 加载完成，共 {len(self.instances)} 个实例")
            print(f"   - 示例 name (最长的10个): {self.sorted_names[:10]}")
    
    def _find_match_in_query(self, name_key: str, query: str) -> Tuple[bool, str, int, int]:
        """
        检查 name 是否出现在 query 中
        
        Args:
            name_key: 标准化后的 name
            query: 用户查询
            
        Returns:
            (is_matched, matched_variant, start_pos, end_pos)
        """
        patterns = self.name_patterns.get(name_key, [])
        
        for pattern, variant in patterns:
            match = pattern.search(query)
            if match:
                return True, variant, match.start(), match.end()
        
        return False, "", -1, -1
    
    def match(self, query: str) -> Tuple[List[str], List[Dict]]:
        """
        在 query 中查找所有匹配的 name
        
        核心逻辑：
        1. 遍历所有 name（按长度降序）
        2. 检查每个 name（及其变体）是否出现在 query 中
        3. 匹配成功后标记位置，避免短 name 重复匹配同一位置
        
        Returns:
            (matched_names, results)
        """
        matched_names = []
        results = []
        matched_ids = set()
        matched_positions = set()  # 已匹配的字符位置
        
        # 遍历所有 name（按长度降序）
        for name_key in self.sorted_names:
            # 检查是否在 query 中
            is_matched, matched_variant, start, end = self._find_match_in_query(name_key, query)
            
            if not is_matched:
                continue
            
            # 检查是否与已匹配位置重叠
            match_positions = set(range(start, end))
            if match_positions & matched_positions:
                continue  # 跳过，已被更长的 name 匹配
            
            # 匹配成功
            matched_names.append(name_key)
            matched_positions.update(match_positions)
            
            # 获取该 name 对应的所有实例
            for instance in self.name_to_instances[name_key]:
                if instance["id"] not in matched_ids:
                    matched_ids.add(instance["id"])
                    results.append({
                        **instance,
                        "matched_name": name_key,
                        "matched_variant": matched_variant,
                        "match_position": (start, end),
                        "score": len(name_key) / len(query),
                    })
        
        # 按 name 长度（分数）排序
        results.sort(key=lambda x: x["score"], reverse=True)
        
        return matched_names, results[:MATCHER_CONFIG["MAX_RESULTS"]]
    
    def retrieve(self, query: str, topk: int = 5) -> List[Dict]:
        """检索接口"""
        matched_names, results = self.match(query)
        
        return [
            {
                "id": r["id"],
                "name": r["name"],
                "class": r["class"],
                "description": r["description"],
                "matched_name": r["matched_name"],
                "score": r["score"],
            }
            for r in results[:topk]
        ]
    
    def get_all_names(self) -> List[str]:
        """获取所有 name"""
        return self.sorted_names
    
    def search_names(self, pattern: str) -> List[str]:
        """搜索匹配模式的 name"""
        regex = re.compile(pattern, re.IGNORECASE)
        return [name for name in self.sorted_names if regex.search(name)]
    
    def check_name_exists(self, name: str) -> bool:
        """检查 name 是否存在于数据库"""
        return self._normalize_key(name) in self.name_to_instances


def main():
    """主函数"""
    
    print("\n" + "=" * 60)
    print("KG 实例名匹配检索器 (基于 name 字段)")
    print("=" * 60)
    
    # 1. 初始化匹配器
    print("\n【1】初始化匹配器")
    matcher = KGNameMatcher(
        kg_file_path=DATABASE_PATH,
        verbose=True
    )
    
    # 打印一些 name 示例
    print("\n   前 30 个 name（按长度排序）:")
    for i, name in enumerate(matcher.sorted_names[:30], 1):
        instances = matcher.name_to_instances[name]
        print(f"      {i:2}. [{name}] -> {len(instances)} 个实例")
    
    # 2. 加载测试问题
    print(f"\n【2】加载测试问题: {BENCHMARK_PATH}")
    with open(BENCHMARK_PATH, 'r', encoding='utf-8') as f:
        benchmark_data = json.load(f)
    print(f"   加载完成: {len(benchmark_data)} 个问题")
    
    # 3. 确保输出目录存在
    output_dir = Path(OUTPUT_PATH).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 4. 执行检索并保存结果
    print("\n" + "=" * 60)
    print("【3】开始检索测试")
    print("=" * 60)
    
    total_questions = len(benchmark_data)
    matched_count = 0
    
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as out_file:
        for idx, item in enumerate(benchmark_data, 1):
            question_id = item.get("id", "N/A")
            query = item.get("question", "")
            
            if not query:
                continue
            
            # 执行匹配
            matched_names, results = matcher.match(query)
            
            if results:
                matched_count += 1
            
            # 写入文件
            out_file.write("=" * 80 + "\n")
            out_file.write(f"ID: {question_id}\n")
            out_file.write(f"Question: {query}\n")
            out_file.write("-" * 80 + "\n")
            out_file.write(f"匹配到的 name: {matched_names}\n")
            out_file.write(f"检索结果数量: {len(results)}\n")
            out_file.write("-" * 80 + "\n")
            
            for i, r in enumerate(results[:10], 1):
                out_file.write(f"[{i}] ID: {r['id']}\n")
                out_file.write(f"    Name: {r['name']}\n")
                out_file.write(f"    Class: {r['class']}\n")
                out_file.write(f"    Matched: {r['matched_name']}")
                if r.get('matched_variant') and r['matched_variant'].lower() != r['matched_name']:
                    out_file.write(f" (via '{r['matched_variant']}')")
                out_file.write("\n")
                out_file.write(f"    Score: {r['score']:.4f}\n")
                if r["description"]: 
                    desc = r['description'][:200] + "..." if len(r['description']) > 200 else r['description']
                    out_file.write(f"    Description: {desc}\n")
                out_file.write("\n")
            
            out_file.write("\n")
            
            # 进度
            if idx % 20 == 0:
                print(f"   已处理 {idx}/{total_questions} 个问题...")
    
    print(f"\n   匹配率: {matched_count}/{total_questions} = {matched_count/total_questions*100:.1f}%")
    print(f"\n结果已保存到: {OUTPUT_PATH}")
    
    # 5. 快速测试
    print("\n" + "=" * 60)
    print("【4】快速测试")
    print("=" * 60)
    
    test_queries = [
        "XTop修setup的时候能size down吗？",
        "如何使用 set_timing_derate 命令？",
        "AOCV 和 OCV 有什么区别？",
        "clock gating 怎么设置？",
        "report_timing 的用法",
        "什么是 EDA？",
        "set_timing_derate命令有哪些选项？",
        "setup violation如何修复？",
        "如何设置 size down 参数？",
    ]
    
    for q in test_queries:
        print(f"\nQuery: {q}")
        matched_names, results = matcher.match(q)
        print(f"  匹配到的 name: {matched_names}")
        print(f"  结果数量: {len(results)}")
        if results:
            for r in results[:3]:
                extra = ""
                if r.get('matched_variant') and r['matched_variant'].lower() != r['matched_name']:
                    extra = f" (via '{r['matched_variant']}')"
                print(f"    - {r['id']}: {r['name']} ({r['class']}){extra}")


if __name__ == "__main__":
    main()