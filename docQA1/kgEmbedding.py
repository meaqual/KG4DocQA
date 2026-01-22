# ========================kgEmbedding.py===========================
import json
import numpy as np
from pathlib import Path
from typing import List, Dict
import torch
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '5'

# ============ 配置 ============
EMBED_MODEL_PATH = "/mnt/public/weights/bge-m3-finetune-v5"
RERANKER_MODEL_PATH = "/mnt/public/weights/bge-reranker-v2-gemma-v5"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

db_path = "/mnt/public/sichuan_a/hyh/queryTest1/qaSchema/xtopDoc/docQA1/dataBase/textContent.json"
query = "如何设置时序约束"

# ============ Embedding ============
class BGEEmbedding:
    def __init__(self, model_path: str = EMBED_MODEL_PATH):
        from transformers import AutoTokenizer, AutoModel
        
        print(f"📦 加载 Embedding: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path).to(DEVICE)
        self.model.eval()
        print("✅ Embedding 加载完成")
    
    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = self.tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                outputs = self.model(**inputs)
                emb = outputs.last_hidden_state[:, 0, :]
                emb = torch.nn.functional.normalize(emb, p=2, dim=1)
                all_embeddings.append(emb.cpu().numpy())
        return np.vstack(all_embeddings)


# ============ Reranker ============
class BGEReranker:
    def __init__(self, model_path: str = RERANKER_MODEL_PATH):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        
        print(f"📦 加载 Reranker: {model_path}")
        # 使用 slow tokenizer 避免 tiktoken 依赖
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, legacy=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(DEVICE)
        self.model.eval()
        print("✅ Reranker 加载完成")
    
    def rerank(self, query: str, docs: List[Dict], topk: int = 5, thresh: float = 0.3) -> List[Dict]:
        if not docs:
            return []
        
        pairs = [[query, d["content"]] for d in docs]
        inputs = self.tokenizer(pairs, padding=True, truncation=True, max_length=512, return_tensors="pt").to(DEVICE)
        
        with torch.no_grad():
            scores = torch.sigmoid(self.model(**inputs).logits[:, 0]).cpu().tolist()
        
        for d, s in zip(docs, scores):
            d["score"] = s
        
        docs = [d for d in docs if d["score"] >= thresh]
        docs.sort(key=lambda x: x["score"], reverse=True)
        return docs[:topk]


# ============ Retriever ============
class KGRetriever:
    def __init__(self, database_path: str):
        with open(database_path, 'r', encoding='utf-8') as f:
            self.database = json.load(f)
        
        self.contents = list(self.database.keys())
        self.ids = list(self.database.values())
        print(f"✅ 加载数据库: {len(self.contents)} 条")
        
        self.embedder = BGEEmbedding()
        self.reranker = BGEReranker()
        
        print("🔨 构建索引...")
        self.index = self.embedder.encode(self.contents)
        print(f"✅ 索引完成: {self.index.shape}")
    
    def retrieve(self, query: str, topk_embed: int = 20, topk_rerank: int = 5) -> List[Dict]:
        q_emb = self.embedder.encode([query])[0]
        scores = np.dot(self.index, q_emb)
        top_idx = np.argsort(scores)[::-1][:topk_embed]
        
        candidates = [{"id": self.ids[i], "content": self.contents[i]} for i in top_idx]
        results = self.reranker.rerank(query, candidates, topk=topk_rerank)
        return results


# ============ 主函数 ============
if __name__ == "__main__":
    import sys
    
    retriever = KGRetriever(db_path)
    results = retriever.retrieve(query)
    
    print(f"\n📌 Query: {query}")
    print("=" * 60)
    for i, r in enumerate(results, 1):
        print(f"[{i}] {r['id']} (score: {r['score']:.4f})")
        print(f"    {r['content'][:100]}...")
        print()