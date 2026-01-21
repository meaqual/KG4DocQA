# ========================kgEmbedding.py===========================
"""
KG Database Retriever - 使用 Embedding + Reranker 检索知识库

Embedding: /mnt/public/weights/bge-m3-finetune-v5
Reranker: /mnt/public/weights/bge-reranker-v2-gemma-v5
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import torch
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '6'

# ============ 全局配置 ============
RETRIEVER_CONFIG = {
    # 模型路径
    "EMBED_MODEL_PATH": "/mnt/public/weights/bge-m3-finetune-v5",
    "RERANKER_MODEL_PATH": "/mnt/public/weights/bge-reranker-v2-gemma-v5",
    
    # 检索参数
    "TOPK_RETRIEVE": 20,           # Embedding 召回数量
    "TOPK_RERANK": 5,              # Reranker 重排后保留数量
    "SCORE_THRESH": 0.3,           # 重排分数阈值
    
    # 设备配置
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "BATCH_SIZE": 32,
}


@dataclass
class RetrievalResult:
    """检索结果"""
    id: str
    content: str
    score: float
    rank: int


class BGEEmbedding:
    """
    BGE-M3 Embedding 模型封装
    """
    
    def __init__(
        self, 
        model_path: str = RETRIEVER_CONFIG["EMBED_MODEL_PATH"],
        device: str = RETRIEVER_CONFIG["DEVICE"],
    ):
        from transformers import AutoTokenizer, AutoModel
        
        print(f"📦 加载 Embedding 模型: {model_path}")
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path).to(device)
        self.model.eval()
        print(f"✅ Embedding 模型加载完成 (device: {device})")
    
    def encode(
        self, 
        texts: List[str], 
        batch_size: int = RETRIEVER_CONFIG["BATCH_SIZE"],
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        编码文本为向量
        
        Args:
            texts: 文本列表
            batch_size: 批处理大小
            show_progress: 是否显示进度
            
        Returns:
            embeddings: shape (n_texts, embedding_dim)
        """
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            if show_progress:
                print(f"   Encoding batch {i // batch_size + 1}/{(len(texts) - 1) // batch_size + 1}")
            
            # Tokenize
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)
            
            # Encode
            with torch.no_grad():
                outputs = self.model(**inputs)
                # 使用 [CLS] token 的输出作为句子表示
                embeddings = outputs.last_hidden_state[:, 0, :]
                # L2 归一化
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                all_embeddings.append(embeddings.cpu().numpy())
        
        return np.vstack(all_embeddings)
    
    def encode_query(self, query: str) -> np.ndarray:
        """编码单个查询"""
        return self.encode([query])[0]


class BGEReranker:
    """
    BGE Reranker 模型封装
    """
    
    def __init__(
        self,
        model_path: str = RETRIEVER_CONFIG["RERANKER_MODEL_PATH"],
        device: str = RETRIEVER_CONFIG["DEVICE"],
    ):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        
        print(f"📦 加载 Reranker 模型: {model_path}")
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
        self.model.eval()
        print(f"✅ Reranker 模型加载完成 (device: {device})")
    
    def rerank(
        self,
        query: str,
        documents: List[Dict[str, str]],  # [{"id": ..., "content": ...}, ...]
        topk: int = RETRIEVER_CONFIG["TOPK_RERANK"],
        score_thresh: float = RETRIEVER_CONFIG["SCORE_THRESH"],
        batch_size: int = RETRIEVER_CONFIG["BATCH_SIZE"],
    ) -> List[Dict]:
        """
        对文档进行重排序
        
        Args:
            query: 查询文本
            documents: 文档列表，每个文档包含 id 和 content
            topk: 返回的文档数量
            score_thresh: 分数阈值
            
        Returns:
            重排序后的文档列表（包含 rerank_score）
        """
        if not documents:
            return []
        
        # 构建 query-document pairs
        pairs = [[query, doc["content"]] for doc in documents]
        
        # 批量计算分数
        all_scores = []
        for i in range(0, len(pairs), batch_size):
            batch_pairs = pairs[i:i + batch_size]
            
            inputs = self.tokenizer(
                batch_pairs,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                # 对于二分类模型，取正类的 logit 或使用 sigmoid
                scores = torch.sigmoid(outputs.logits[:, 0]).cpu().numpy()
                all_scores.extend(scores.tolist())
        
        # 添加分数到文档
        scored_docs = []
        for doc, score in zip(documents, all_scores):
            doc_with_score = doc.copy()
            doc_with_score["rerank_score"] = score
            scored_docs.append(doc_with_score)
        
        # 过滤低分文档
        filtered_docs = [
            doc for doc in scored_docs 
            if doc["rerank_score"] >= score_thresh
        ]
        
        # 按分数降序排序
        filtered_docs.sort(key=lambda x: x["rerank_score"], reverse=True)
        
        return filtered_docs[:topk]


class KGDatabaseRetriever:
    """
    KG 数据库检索器
    
    两阶段检索：Embedding 召回 + Reranker 精排
    """
    
    def __init__(
        self,
        database_path: str,
        embed_model_path: str = RETRIEVER_CONFIG["EMBED_MODEL_PATH"],
        reranker_model_path: str = RETRIEVER_CONFIG["RERANKER_MODEL_PATH"],
        device: str = RETRIEVER_CONFIG["DEVICE"],
        build_index: bool = True,
    ):
        self.database_path = database_path
        self.device = device
        
        # 加载数据库
        self.database = self._load_database(database_path)
        self.id_list = list(self.database.keys())
        self.content_list = list(self.database.values())
        print(f"✅ 加载数据库: {len(self.database)} 条记录")
        
        # 初始化 Embedding 模型
        self.embedder = BGEEmbedding(model_path=embed_model_path, device=device)
        
        # 初始化 Reranker 模型
        self.reranker = BGEReranker(model_path=reranker_model_path, device=device)
        
        # 构建索引
        if build_index:
            self.index = self._build_index()
        else:
            self.index = None
    
    def _load_database(self, path: str) -> Dict[str, str]:
        """加载 KG 数据库"""
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _build_index(self) -> np.ndarray:
        """
        构建向量索引
        
        Returns:
            embeddings: shape (n_docs, embedding_dim)
        """
        print("🔨 构建向量索引...")
        
        # 过滤空内容
        valid_contents = [c for c in self.content_list if c.strip()]
        
        # 编码所有文档
        embeddings = self.embedder.encode(valid_contents, show_progress=True)
        
        print(f"✅ 索引构建完成: {embeddings.shape}")
        return embeddings
    
    def save_index(self, save_path: str):
        """保存索引到文件"""
        np.save(save_path, self.index)
        print(f"✅ 索引已保存: {save_path}")
    
    def load_index(self, load_path: str):
        """从文件加载索引"""
        self.index = np.load(load_path)
        print(f"✅ 索引已加载: {self.index.shape}")
    
    def retrieve(
        self,
        query: str,
        topk_retrieve: int = RETRIEVER_CONFIG["TOPK_RETRIEVE"],
        topk_rerank: int = RETRIEVER_CONFIG["TOPK_RERANK"],
        score_thresh: float = RETRIEVER_CONFIG["SCORE_THRESH"],
    ) -> List[RetrievalResult]:
        """
        两阶段检索
        
        1. Embedding 召回 topk_retrieve 条
        2. Reranker 精排，返回 topk_rerank 条
        
        Args:
            query: 查询文本
            topk_retrieve: Embedding 召回数量
            topk_rerank: Reranker 返回数量
            score_thresh: 分数阈值
            
        Returns:
            检索结果列表
        """
        print(f"\n{'='*60}")
        print(f"📌 Query: {query}")
        
        # ========== Stage 1: Embedding 召回 ==========
        query_embedding = self.embedder.encode_query(query)
        
        # 计算余弦相似度 (由于已归一化，直接点积)
        similarities = np.dot(self.index, query_embedding)
        
        # 获取 topk 索引
        topk_indices = np.argsort(similarities)[::-1][:topk_retrieve]
        
        print(f"   Stage 1 - Embedding 召回: {len(topk_indices)} 条")
        
        # 构建候选文档
        candidates = []
        for idx in topk_indices:
            if self.content_list[idx].strip():  # 跳过空内容
                candidates.append({
                    "id": self.id_list[idx],
                    "content": self.content_list[idx],
                    "embed_score": float(similarities[idx]),
                })
        
        # ========== Stage 2: Reranker 精排 ==========
        reranked_docs = self.reranker.rerank(
            query=query,
            documents=candidates,
            topk=topk_rerank,
            score_thresh=score_thresh,
        )
        
        print(f"   Stage 2 - Reranker 精排: {len(reranked_docs)} 条")
        
        # 构建返回结果
        results = []
        for rank, doc in enumerate(reranked_docs, 1):
            results.append(RetrievalResult(
                id=doc["id"],
                content=doc["content"],
                score=doc["rerank_score"],
                rank=rank,
            ))
        
        return results
    
    def batch_retrieve(
        self,
        queries: List[str],
        topk_retrieve: int = RETRIEVER_CONFIG["TOPK_RETRIEVE"],
        topk_rerank: int = RETRIEVER_CONFIG["TOPK_RERANK"],
    ) -> Dict[str, List[RetrievalResult]]:
        """批量检索"""
        results = {}
        for query in queries:
            results[query] = self.retrieve(query, topk_retrieve, topk_rerank)
        return results


def print_results(results: List[RetrievalResult]):
    """打印检索结果"""
    if not results:
        print("   ❌ 无检索结果")
        return
        
    for r in results:
        print(f"\n  [{r.rank}] {r.id}")
        print(f"      Score: {r.score:.4f}")
        content_preview = r.content[:150] + "..." if len(r.content) > 150 else r.content
        print(f"      Content: {content_preview}")


# ============ 测试用例 ============

# 示例查询（EDA/芯片设计领域）
SAMPLE_QUERIES = [
    "如何设置时序约束的最大转换时间",
    "report_timing 命令怎么用",
    "什么是 setup slack",
    "如何优化 hold time violation",
    "clock skew 怎么处理",
    "set_max_fanout 的用法",
    "如何查看关键路径",
    "时钟树综合的基本流程",
]


def create_sample_database(output_path: str):
    """创建示例数据库（用于测试）"""
    sample_db = {
        "kg_Command_0001": "set_max_transition value [-clock] [-data] | 设置最大转换时间约束，用于控制信号上升/下降时间 | 应用场景: 时序约束设置",
        "kg_Command_0002": "report_timing [-from] [-to] [-max_paths n] | 报告时序路径信息，显示关键路径的详细时序分析结果",
        "kg_Command_0003": "set_max_fanout value object_list | 设置最大扇出约束，限制单个驱动器驱动的负载数量",
        "kg_Command_0004": "report_clock_timing | 报告时钟路径的时序信息，包括时钟延迟和偏斜",
        "kg_Command_0005": "set_clock_uncertainty value | 设置时钟不确定性，包括抖动和偏斜的裕量",
        "kg_Concept_0001": "setup slack 表示数据信号到达时间与时钟边沿之间的裕量，正值表示满足时序要求，负值表示时序违规",
        "kg_Concept_0002": "hold time violation 表示数据保持时间不足，数据在时钟边沿后变化过快，需要增加延迟来修复",
        "kg_Concept_0003": "clock skew 是时钟信号到达不同寄存器的时间差异，过大会导致时序问题，需要通过时钟树综合来优化",
        "kg_Concept_0004": "关键路径 (critical path) 是设计中时序裕量最小的路径，决定了芯片的最高工作频率",
        "kg_Concept_0005": "时钟树综合 (CTS) 是将时钟信号均匀分布到所有时序单元的过程，目标是最小化时钟偏斜",
        "kg_Flow_0001": "时钟树综合基本流程：1. 定义时钟源 2. 设置时钟约束 3. 构建时钟树 4. 平衡时钟延迟 5. 验证时钟质量",
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sample_db, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 已创建示例数据库: {output_path}")
    return output_path


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="KG Database Retriever")
    parser.add_argument("--database", type=str, default="kg_database.json", help="数据库路径")
    parser.add_argument("--query", type=str, default=None, help="单个查询")
    parser.add_argument("--topk_retrieve", type=int, default=RETRIEVER_CONFIG["TOPK_RETRIEVE"], help="Embedding 召回数量")
    parser.add_argument("--topk_rerank", type=int, default=RETRIEVER_CONFIG["TOPK_RERANK"], help="Reranker 返回数量")
    parser.add_argument("--create_sample", action="store_true", help="创建示例数据库")
    
    args = parser.parse_args()
    
    # 更新全局配置
    RETRIEVER_CONFIG["TOPK_RETRIEVE"] = args.topk_retrieve
    RETRIEVER_CONFIG["TOPK_RERANK"] = args.topk_rerank
    
    database_path = args.database
    
    # 检查/创建数据库
    if args.create_sample or not Path(database_path).exists():
        database_path = create_sample_database(database_path)
    
    # 初始化检索器
    print("\n" + "=" * 60)
    print("🚀 初始化 KG Database Retriever")
    print("=" * 60)
    
    retriever = KGDatabaseRetriever(
        database_path=database_path,
        embed_model_path=RETRIEVER_CONFIG["EMBED_MODEL_PATH"],
        reranker_model_path=RETRIEVER_CONFIG["RERANKER_MODEL_PATH"],
    )
    
    # 执行检索
    print("\n" + "=" * 60)
    print("🔍 开始检索")
    print("=" * 60)
    
    if args.query:
        # 单个查询
        queries = [args.query]
    else:
        # 使用示例查询
        queries = SAMPLE_QUERIES
    
    for query in queries:
        results = retriever.retrieve(
            query=query,
            topk_retrieve=args.topk_retrieve,
            topk_rerank=args.topk_rerank,
        )
        print_results(results)
        print()


if __name__ == "__main__":
    main()