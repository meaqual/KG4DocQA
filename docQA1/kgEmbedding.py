# ========================kgEmbedding.py===========================

import json
import numpy as np
from typing import List, Dict
import torch
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '5'

# ============ 配置 ============
EMBED_MODEL_PATH = "/mnt/public/weights/bge-m3-finetune-v5"
RERANKER_BASE_MODEL = "/mnt/public/weights/recover_models/bge-reranker-v2-gemma"
RERANKER_ADAPTER_PATH = "/mnt/public/weights/bge-reranker-v2-gemma-v5"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 数据库路径
DATABASE_PATH = "/mnt/public/sichuan_a/hyh/queryTest1/qaSchema/xtopDoc/docQA1/dataBase/textContent.json"
# 测试问题路径
BENCHMARK_PATH = "/mnt/public/sichuan_a/hyh/queryTest1/qaSchema/xtopDoc/docQA1/testData/gt_benchmark.json"
# 输出结果路径
OUTPUT_PATH = "/mnt/public/sichuan_a/hyh/queryTest1/qaSchema/xtopDoc/docQA1/results/kgEmbedding_results.txt"


# ============ Embedding ============
class BGEEmbedding:
    """BGE Embedding 模型封装"""
    
    def __init__(self, model_path: str = EMBED_MODEL_PATH):
        from transformers import AutoTokenizer, AutoModel
        
        print(f"加载 Embedding: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path).to(DEVICE)
        self.model.eval()
        print("Embedding 加载完成")
    
    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """编码文本为向量"""
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = self.tokenizer(
                batch, 
                padding=True, 
                truncation=True, 
                max_length=512, 
                return_tensors="pt"
            ).to(DEVICE)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                emb = outputs.last_hidden_state[:, 0, :]
                emb = torch.nn.functional.normalize(emb, p=2, dim=1)
                all_embeddings.append(emb.cpu().numpy())
        
        return np.vstack(all_embeddings)


# ============ Reranker ============
class BGEReranker:
    """BGE Reranker 模型封装 - 支持 LoRA Adapter"""
    
    def __init__(
        self, 
        base_model_path: str = RERANKER_BASE_MODEL,
        adapter_path: str = RERANKER_ADAPTER_PATH
    ):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        from peft import PeftModel
        
        print(f"加载 Reranker 基础模型: {base_model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(adapter_path, use_fast=False, legacy=True)
        
        # 加载基础模型
        base_model = AutoModelForSequenceClassification.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
        )
        
        # 加载 LoRA adapter
        print(f"加载 LoRA Adapter: {adapter_path}")
        self.model = PeftModel.from_pretrained(base_model, adapter_path)
        self.model = self.model.to(DEVICE)
        self.model.eval()
        print("Reranker 加载完成")
    
    def rerank_and_filter(
        self, 
        query: str, 
        docs: List[Dict], 
        topk: int = 5, 
        score_thresh: float = 0.3
    ) -> List[Dict]:
        """对候选文档进行重排序和过滤"""
        if not docs:
            return []
        
        pairs = [[query, d["content"]] for d in docs]
        
        inputs = self.tokenizer(
            pairs, 
            padding=True, 
            truncation=True, 
            max_length=512, 
            return_tensors="pt"
        ).to(DEVICE)
        
        with torch.no_grad():
            logits = self.model(**inputs).logits
            scores = torch.sigmoid(logits[:, 0]).cpu().tolist()
        
        for doc, score in zip(docs, scores):
            doc["score"] = score
        
        filtered_docs = [d for d in docs if d["score"] >= score_thresh]
        filtered_docs.sort(key=lambda x: x["score"], reverse=True)
        
        return filtered_docs[:topk]


# ============ Retriever ============
class KGRetriever:
    """知识图谱检索器"""
    
    def __init__(self, database_path: str, topk: int = 20):
        print(f"加载知识库: {database_path}")
        with open(database_path, 'r', encoding='utf-8') as f:
            self.database = json.load(f)
        
        self.contents = list(self.database.keys())
        self.ids = list(self.database.values())
        print(f"知识库加载完成: {len(self.contents)} 条")

        self.topk = topk
        self.embedder = BGEEmbedding()
        self.reranker = BGEReranker()
        
        print("构建向量索引...")
        self.index = self.embedder.encode(self.contents)
        print(f"索引完成: {self.index.shape}")
    
    def get_relevant_documents(self, query: str, topk: int = None) -> List[Dict]:
        """检索相关文档"""
        topk = topk or self.topk
        
        query_embedding = self.embedder.encode([query])[0]
        scores = np.dot(self.index, query_embedding)
        top_indices = np.argsort(scores)[::-1][:topk]
        
        candidates = [
            {
                "id": self.ids[i], 
                "content": self.contents[i],
                "embed_score": float(scores[i])
            } 
            for i in top_indices
        ]
        
        return candidates
    
    def retrieve(
        self, 
        query: str, 
        topk_embed: int = 20, 
        topk_rerank: int = 5,
        score_thresh: float = 0.3
    ) -> List[Dict]:
        """完整的检索流程：召回 + 重排序"""
        candidates = self.get_relevant_documents(query, topk=topk_embed)
        results = self.reranker.rerank_and_filter(
            query, 
            candidates, 
            topk=topk_rerank,
            score_thresh=score_thresh
        )
        return results


# ============ 主函数 ============
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("初始化 KG Retriever")
    print("=" * 60)
    
    retriever = KGRetriever(
        database_path=DATABASE_PATH,
        topk=20
    )
    
    # 加载测试问题
    print(f"\n加载测试问题: {BENCHMARK_PATH}")
    with open(BENCHMARK_PATH, 'r', encoding='utf-8') as f:
        benchmark_data = json.load(f)
    print(f"加载完成: {len(benchmark_data)} 个问题")
    
    print("\n" + "=" * 60)
    print("开始检索测试")
    print("=" * 60)
    
    # 打开输出文件
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as out_file:
        for item in benchmark_data:
            question_id = item.get("id", "N/A")
            query = item.get("question", "")
            
            if not query:
                continue
            
            results = retriever.retrieve(
                query=query,
                topk_embed=20,
                topk_rerank=5,
                score_thresh=0.3
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
                out_file.write(f"[{i}] ID: {r['id']} | Score: {r['score']:.4f}\n")
                out_file.write(f"    Content: {r['content']}\n")
                out_file.write("\n")
            
            out_file.write("\n")
    
    print("\n" + "=" * 60)
    print(f"结果已保存到: {OUTPUT_PATH}")
    print("=" * 60)