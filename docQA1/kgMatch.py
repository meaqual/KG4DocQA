# =========================kgMatch.py=============================
"""
KG 实例名匹配检索器 - 基于正则/字符串匹配

无需向量数据库，直接匹配 query 中是否包含 KG 实例名
"""

import os
import re
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict

os.environ['CUDA_VISIBLE_DEVICES'] = '5'

# ============ 路径配置 ============
# 数据库路径
DATABASE_PATH = "/mnt/public/sichuan_a/hyh/queryTest1/qaSchema/xtopDoc/docQA1/dataBase/textContent.json"
# 测试问题路径
BENCHMARK_PATH = "/mnt/public/sichuan_a/hyh/queryTest1/qaSchema/xtopDoc/docQA1/testData/gt_benchmark.json"
# 输出结果路径
OUTPUT_PATH = "/mnt/public/sichuan_a/hyh/queryTest1/qaSchema/xtopDoc/docQA1/results/kgMatch_results.txt"


# ============ 匹配器配置 ============
MATCHER_CONFIG = {
    # 匹配模式: "regex" | "exact" | "fuzzy"
    "MATCH_MODE": "regex",
    
    # 正则匹配选项
    "CASE_SENSITIVE": False,      # 是否区分大小写
    "WORD_BOUNDARY": False,       # 是否使用单词边界 \b（中文场景建议 False）
    
    # 模糊匹配选项（fuzzy 模式）
    "FUZZY_THRESHOLD": 0.8,       # 模糊匹配阈值
    
    # 结果过滤
    "MIN_NAME_LENGTH": 2,         # 最小实例名长度（过滤太短的名字）
    "MAX_RESULTS": 50,            # 最大返回结果数
    
    # 优先级权重（用于排序）
    "PRIORITY_WEIGHTS": {
        "exact": 1.0,             # 完全匹配
        "case_insensitive": 0.9,  # 大小写不敏感匹配
        "partial": 0.7,           # 部分匹配
        "fuzzy": 0.6,             # 模糊匹配
    },
}


@dataclass
class MatchResult:
    """匹配结果"""
    id: str                       # 实例 ID
    name: str                     # 匹配到的实例名
    content: str                  # 实例内容
    match_type: str               # 匹配类型: exact | case_insensitive | partial | fuzzy
    match_position: Tuple[int, int]  # 匹配位置 (start, end)
    score: float                  # 匹配分数
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "name": self.name,
            "content": self.content,
            "match_type": self.match_type,
            "match_position": self.match_position,
            "score": self.score,
        }


@dataclass 
class KGInstance:
    """KG 实例"""
    id: str
    content: str
    names: List[str] = field(default_factory=list)  # 可能有多个名字/别名


class KGNameExtractor:
    """
    从 KG 内容中提取实例名称
    """
    
    @classmethod
    def extract_names(cls, instance_id: str, content: str) -> List[str]:
        """
        从实例内容中提取名称
        
        Args:
            instance_id: 实例 ID
            content: 实例内容
            
        Returns:
            names: 提取到的名称列表
        """
        names = set()
        content_stripped = content.strip()
        
        # 1. 从 ID 中提取（如果 ID 包含有意义的名字）
        # 例如: kg_Command_set_max_transition -> set_max_transition
        id_parts = instance_id.split('_')
        if len(id_parts) > 2:
            potential_name = '_'.join(id_parts[2:])
            if len(potential_name) >= MATCHER_CONFIG["MIN_NAME_LENGTH"]:
                names.add(potential_name)
        
        # 2. 提取命令名（英文下划线格式）
        # 例如: "set_max_transition value" -> set_max_transition
        cmd_match = re.match(r'^([a-z_][a-z0-9_]*)', content_stripped, re.IGNORECASE)
        if cmd_match:
            name = cmd_match.group(1)
            if len(name) >= MATCHER_CONFIG["MIN_NAME_LENGTH"]:
                names.add(name)
        
        # 3. 提取中文术语名（开头的中文词）
        chinese_match = re.match(r'^([\u4e00-\u9fa5]{2,15})', content_stripped)
        if chinese_match:
            names.add(chinese_match.group(1))
        
        # 4. 提取括号中的英文术语
        # 例如: "保持时间违规 (hold time violation)" -> hold time violation
        paren_terms = re.findall(r'\(([a-zA-Z][a-zA-Z0-9_\s\-]{1,40})\)', content)
        for term in paren_terms:
            term = term.strip()
            if len(term) >= MATCHER_CONFIG["MIN_NAME_LENGTH"]:
                names.add(term)
        
        # 5. 提取缩写（全大写）
        # 例如: "WNS (Worst Negative Slack)" -> WNS
        abbr_terms = re.findall(r'\b([A-Z]{2,6})\b', content)
        for term in abbr_terms:
            names.add(term)
        
        # 6. 提取 pipe 分隔符前的内容
        # 例如: "set_max_fanout value | 描述" -> set_max_fanout
        if '|' in content_stripped:
            before_pipe = content_stripped.split('|')[0].strip()
            first_word = before_pipe.split()[0] if before_pipe else None
            if first_word and len(first_word) >= MATCHER_CONFIG["MIN_NAME_LENGTH"]:
                names.add(first_word)
        
        return list(names)


class KGInstanceMatcher:
    """
    KG 实例名匹配器
    
    通过正则/字符串匹配检测 query 中是否包含 KG 实例名
    """
    
    def __init__(
        self,
        kg_data: Dict[str, str] = None,
        kg_file_path: str = None,
        config: Dict = None,
        verbose: bool = True,
    ):
        """
        初始化匹配器
        
        Args:
            kg_data: KG 数据字典 {content: id} 或 {id: content}
            kg_file_path: KG 数据文件路径
            config: 配置覆盖
            verbose: 是否打印详细信息
        """
        self.verbose = verbose
        
        # 更新配置
        if config:
            MATCHER_CONFIG.update(config)
        
        # 加载数据
        if kg_data:
            self.kg_data = kg_data
        elif kg_file_path:
            self.kg_data = self._load_kg_file(kg_file_path)
        else:
            raise ValueError("必须提供 kg_data 或 kg_file_path")
        
        # 转换数据格式为 {id: content}
        self.kg_data = self._normalize_data(self.kg_data)
        
        # 构建实例索引
        self.instances: List[KGInstance] = []
        self.name_to_instance: Dict[str, List[KGInstance]] = defaultdict(list)
        self._build_index()
        
        if self.verbose:
            print(f"KG 实例匹配器初始化完成")
            print(f"   - 实例数量: {len(self.instances)}")
            print(f"   - 名称数量: {len(self.name_to_instance)}")
    
    def _load_kg_file(self, path: str) -> Dict[str, str]:
        """加载 KG 文件"""
        if self.verbose:
            print(f"加载 KG 文件: {path}")
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _normalize_data(self, data: Dict[str, str]) -> Dict[str, str]:
        """
        标准化数据格式为 {id: content}
        
        输入可能是:
        - {content: id} 格式 (buildDatabase.py 输出)
        - {id: content} 格式
        """
        if not data:
            return {}
        
        # 检查第一个 key 是否像 ID（以 kg_ 或数字开头）
        first_key = next(iter(data.keys()))
        first_value = data[first_key]
        
        # 如果 key 看起来像内容（较长的文本），value 看起来像 ID
        # 则需要翻转
        if len(first_key) > 50 or (
            isinstance(first_value, str) and 
            (first_value.startswith("kg_") or first_value[0].isdigit())
        ):
            if self.verbose:
                print("   检测到 {content: id} 格式，正在转换...")
            # 翻转：{content: id} -> {id: content}
            normalized = {}
            for content, ids in data.items():
                # ids 可能是 "id1, id2, id3" 格式
                for id_str in ids.split(", "):
                    id_str = id_str.strip()
                    if id_str:
                        normalized[id_str] = content
            return normalized
        
        return data
    
    def _build_index(self):
        """构建名称索引"""
        if self.verbose:
            print("🔨 构建名称索引...")
        
        for instance_id, content in self.kg_data.items():
            # 提取名称
            names = KGNameExtractor.extract_names(instance_id, content)
            
            # 创建实例对象
            instance = KGInstance(
                id=instance_id,
                content=content,
                names=names,
            )
            self.instances.append(instance)
            
            # 建立名称到实例的映射
            for name in names:
                name_lower = name.lower()
                self.name_to_instance[name_lower].append(instance)
        
        # 按名称长度降序排序（优先匹配长名称，避免短名称误匹配）
        self.sorted_names = sorted(
            self.name_to_instance.keys(),
            key=len,
            reverse=True
        )
        
        if self.verbose:
            total_names = sum(len(inst.names) for inst in self.instances)
            print(f"   - 提取到 {total_names} 个名称")
    
    def _compile_pattern(self, name: str) -> re.Pattern:
        """编译正则表达式"""
        escaped_name = re.escape(name)
        
        if MATCHER_CONFIG["WORD_BOUNDARY"]:
            pattern = rf'\b{escaped_name}\b'
        else:
            pattern = escaped_name
        
        flags = 0 if MATCHER_CONFIG["CASE_SENSITIVE"] else re.IGNORECASE
        return re.compile(pattern, flags)
    
    def match_regex(self, query: str) -> List[MatchResult]:
        """
        正则匹配模式
        """
        results = []
        matched_positions = set()
        
        for name in self.sorted_names:
            if len(name) < MATCHER_CONFIG["MIN_NAME_LENGTH"]:
                continue
            
            pattern = self._compile_pattern(name)
            
            for match in pattern.finditer(query):
                start, end = match.span()
                
                # 检查是否与已匹配位置重叠
                overlap = False
                for pos in matched_positions:
                    if not (end <= pos[0] or start >= pos[1]):
                        overlap = True
                        break
                
                if overlap:
                    continue
                
                matched_positions.add((start, end))
                
                # 确定匹配类型
                matched_text = match.group()
                if matched_text == name:
                    match_type = "exact"
                elif matched_text.lower() == name.lower():
                    match_type = "case_insensitive"
                else:
                    match_type = "partial"
                
                score = MATCHER_CONFIG["PRIORITY_WEIGHTS"].get(match_type, 0.5)
                
                for instance in self.name_to_instance[name.lower()]:
                    results.append(MatchResult(
                        id=instance.id,
                        name=name,
                        content=instance.content,
                        match_type=match_type,
                        match_position=(start, end),
                        score=score,
                    ))
        
        return results
    
    def match_exact(self, query: str) -> List[MatchResult]:
        """
        精确匹配模式（简单的 in 操作）
        """
        results = []
        query_lower = query.lower()
        
        for name in self.sorted_names:
            if len(name) < MATCHER_CONFIG["MIN_NAME_LENGTH"]:
                continue
            
            name_lower = name.lower()
            
            start = 0
            while True:
                pos = query_lower.find(name_lower, start)
                if pos == -1:
                    break
                
                matched_text = query[pos:pos + len(name)]
                if matched_text == name:
                    match_type = "exact"
                    score = 1.0
                else:
                    match_type = "case_insensitive"
                    score = 0.9
                
                for instance in self.name_to_instance[name_lower]:
                    results.append(MatchResult(
                        id=instance.id,
                        name=name,
                        content=instance.content,
                        match_type=match_type,
                        match_position=(pos, pos + len(name)),
                        score=score,
                    ))
                
                start = pos + 1
        
        return results
    
    def match_fuzzy(self, query: str) -> List[MatchResult]:
        """
        模糊匹配模式（基于编辑距离）
        """
        from difflib import SequenceMatcher
        
        results = []
        query_words = query.lower().split()
        threshold = MATCHER_CONFIG["FUZZY_THRESHOLD"]
        
        for name in self.sorted_names:
            if len(name) < MATCHER_CONFIG["MIN_NAME_LENGTH"]:
                continue
            
            name_lower = name.lower()
            
            for word in query_words:
                ratio = SequenceMatcher(None, word, name_lower).ratio()
                
                if ratio >= threshold:
                    pos = query.lower().find(word)
                    
                    for instance in self.name_to_instance[name_lower]:
                        results.append(MatchResult(
                            id=instance.id,
                            name=name,
                            content=instance.content,
                            match_type="fuzzy",
                            match_position=(pos, pos + len(word)),
                            score=ratio,
                        ))
        
        return results
    
    def match(
        self, 
        query: str, 
        mode: str = None,
        deduplicate: bool = True,
    ) -> List[MatchResult]:
        """
        执行匹配
        
        Args:
            query: 查询文本
            mode: 匹配模式，默认使用配置中的模式
            deduplicate: 是否去重
            
        Returns:
            匹配结果列表（按分数降序）
        """
        mode = mode or MATCHER_CONFIG["MATCH_MODE"]
        
        if mode == "regex":
            results = self.match_regex(query)
        elif mode == "exact":
            results = self.match_exact(query)
        elif mode == "fuzzy":
            results = self.match_fuzzy(query)
        else:
            raise ValueError(f"未知的匹配模式: {mode}")
        
        # 去重（同一实例只保留最高分）
        if deduplicate:
            seen = {}
            for r in results:
                if r.id not in seen or r.score > seen[r.id].score:
                    seen[r.id] = r
            results = list(seen.values())
        
        # 按分数降序排序
        results.sort(key=lambda x: x.score, reverse=True)
        
        return results[:MATCHER_CONFIG["MAX_RESULTS"]]
    
    def retrieve(
        self,
        query: str,
        topk: int = 5,
        mode: str = None,
    ) -> List[Dict]:
        """
        检索接口（与 kgEmbedding 保持一致）
        
        Args:
            query: 查询文本
            topk: 返回结果数量
            mode: 匹配模式
            
        Returns:
            结果列表，每个元素包含 id, content, score
        """
        results = self.match(query, mode=mode)
        
        return [
            {
                "id": r.id,
                "content": r.content,
                "score": r.score,
                "match_type": r.match_type,
                "matched_name": r.name,
            }
            for r in results[:topk]
        ]
    
    def batch_match(
        self, 
        queries: List[str],
        mode: str = None,
    ) -> Dict[str, List[MatchResult]]:
        """批量匹配"""
        return {q: self.match(q, mode) for q in queries}
    
    def get_all_names(self) -> List[str]:
        """获取所有实例名称"""
        return list(self.name_to_instance.keys())
    
    def get_instance_by_name(self, name: str) -> List[KGInstance]:
        """根据名称获取实例"""
        return self.name_to_instance.get(name.lower(), [])
    
    def search_names(self, pattern: str) -> List[str]:
        """搜索匹配模式的名称"""
        regex = re.compile(pattern, re.IGNORECASE)
        return [name for name in self.sorted_names if regex.search(name)]


def print_match_results(query: str, results: List[MatchResult], max_show: int = 10):
    """打印匹配结果"""
    print(f"\n{'='*70}")
    print(f"Query: {query}")
    print(f"   匹配数量: {len(results)}")
    
    if not results:
        print("   无匹配结果")
        return
    
    for i, r in enumerate(results[:max_show], 1):
        print(f"\n  [{i}] {r.id}")
        print(f"      名称: {r.name}")
        print(f"      类型: {r.match_type} | 分数: {r.score:.2f}")
        print(f"      位置: {r.match_position}")
        content_preview = r.content[:80] + "..." if len(r.content) > 80 else r.content
        print(f"      内容: {content_preview}")
    
    if len(results) > max_show:
        print(f"\n   ... 还有 {len(results) - max_show} 个结果")


# ============ 主函数 ============
def main():
    """主函数 - 读取测试数据并输出结果"""
    
    print("\n" + "=" * 60)
    print("KG 实例名匹配检索器")
    print("=" * 60)
    
    # ========== 1. 初始化匹配器 ==========
    print("\n【1】初始化匹配器")
    print(f"   数据库路径: {DATABASE_PATH}")
    
    matcher = KGInstanceMatcher(
        kg_file_path=DATABASE_PATH,
        verbose=True
    )
    
    # ========== 2. 加载测试问题 ==========
    print(f"\n【2】加载测试问题: {BENCHMARK_PATH}")
    with open(BENCHMARK_PATH, 'r', encoding='utf-8') as f:
        benchmark_data = json.load(f)
    print(f"   加载完成: {len(benchmark_data)} 个问题")
    
    # ========== 3. 确保输出目录存在 ==========
    output_dir = Path(OUTPUT_PATH).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ========== 4. 执行检索并保存结果 ==========
    print("\n" + "=" * 60)
    print("【3】开始检索测试")
    print("=" * 60)
    
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as out_file:
        for item in benchmark_data:
            question_id = item.get("id", "N/A")
            query = item.get("question", "")
            
            if not query:
                continue
            
            # 执行检索
            results = matcher.retrieve(
                query=query,
                topk=5,
                mode="regex"
            )
            
            # 写入文件
            out_file.write("=" * 80 + "\n")
            out_file.write(f"ID: {question_id}\n")
            out_file.write(f"Question: {query}\n")
            out_file.write("-" * 80 + "\n")
            out_file.write(f"检索结果数量: {len(results)}\n")
            out_file.write(f"结果ID列表: {[r['id'] for r in results]}\n")
            out_file.write("-" * 80 + "\n")
            
            for i, r in enumerate(results, 1):
                out_file.write(f"[{i}] ID: {r['id']} | Score: {r['score']:.4f} | Match: {r['match_type']}\n")
                out_file.write(f"    Matched Name: {r['matched_name']}\n")
                out_file.write(f"    Content: {r['content']}\n")
                out_file.write("\n")
            
            out_file.write("\n")
        
    
    print("\n" + "=" * 60)
    print(f"结果已保存到: {OUTPUT_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()