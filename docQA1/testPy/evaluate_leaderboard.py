# ========================evaluate_leaderboard.py===========================

import json
import numpy as np
from typing import List, Dict, Set, Optional, Tuple
import torch
import os
from dataclasses import dataclass
from datetime import datetime
from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = '5'

# ============ 配置 ============
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 模型路径配置
MODELS_CONFIG = {
    "embedding": {
        "bge-m3": "/mnt/public/weights/recover_models/bge-m3",
        "bge-m3-finetuned": "/mnt/public/weights/recover_models/bge-m3-finetune-v5",
    },
    "reranker": {
        "bge-reranker-v2-gemma": {
            "base": "/mnt/public/weights/recover_models/bge-reranker-v2-gemma",
            "adapter": None
        },
        "bge-reranker-v2-gemma-finetuned": {
            "base": "/mnt/public/weights/recover_models/bge-reranker-v2-gemma",
            "adapter": "/mnt/public/weights/recover_models/bge-reranker-v2-gemma-v5"
        },
    }
}

# 数据路径
DATABASE_PATH = "/mnt/public/sichuan_a/hyh/queryTest1/qaSchema/xtopDoc/docQA1/dataBase/textContent.json"
BENCHMARK_PATH = "/mnt/public/sichuan_a/hyh/queryTest1/qaSchema/xtopDoc/docQA1/testData/gt_benchmark.json"
KG_TO_CHUNK_PATH = "/mnt/public/sichuan_a/hyh/queryTest1/qaSchema/xtopDoc/docQA1/dataBase/textWithId.json"
OUTPUT_DIR = "/mnt/public/sichuan_a/hyh/queryTest1/qaSchema/xtopDoc/docQA1/results/exp3"

# 检索结果保存的 Top-K
SAVE_TOPK = 20


# ============ Embedding Models ============
class BGEEmbedding:
    """BGE Embedding 模型封装"""
    
    def __init__(self, model_path: str):
        from transformers import AutoTokenizer, AutoModel
        
        print(f"  加载 Embedding: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path).to(DEVICE)
        self.model.eval()
    
    def encode(self, texts: List[str], batch_size: int = 32, show_progress: bool = False) -> np.ndarray:
        """编码文本为向量"""
        all_embeddings = []
        
        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="    编码文本", leave=False)
        
        for i in iterator:
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
    
    def release(self):
        """释放模型资源"""
        del self.model
        del self.tokenizer
        torch.cuda.empty_cache()


# ============ Reranker Models ============
class BGEReranker:
    """BGE Reranker 模型封装"""
    
    def __init__(self, base_model_path: str, adapter_path: Optional[str] = None):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        
        print(f"  加载 Reranker: {base_model_path}")
        
        if adapter_path:
            from peft import PeftModel
            print(f"  加载 Adapter: {adapter_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(adapter_path, use_fast=False, legacy=True)
            base_model = AutoModelForSequenceClassification.from_pretrained(
                base_model_path,
                torch_dtype=torch.float16,
            )
            self.model = PeftModel.from_pretrained(base_model, adapter_path)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=False, legacy=True)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                base_model_path,
                torch_dtype=torch.float16,
            )
        
        self.model = self.model.to(DEVICE)
        self.model.eval()
    
    def rerank(self, query: str, docs: List[Dict]) -> List[Dict]:
        """对候选文档进行重排序"""
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
            doc["rerank_score"] = score
        
        docs.sort(key=lambda x: x["rerank_score"], reverse=True)
        return docs
    
    def release(self):
        """释放模型资源"""
        del self.model
        del self.tokenizer
        torch.cuda.empty_cache()


# ============ 数据加载 ============
class DataLoader:
    """数据加载器"""
    
    def __init__(self):
        print("加载数据...")
        
        # 加载知识库: {content: kg_id}
        with open(DATABASE_PATH, 'r', encoding='utf-8') as f:
            self.database = json.load(f)
        
        # 构建索引列表：contents[i] 对应 kg_ids[i]
        self.contents = list(self.database.keys())
        self.kg_ids = list(self.database.values())
        print(f"  知识库: {len(self.contents)} 条内容")
        
        # 加载 KG ID 到 Chunk ID 映射
        with open(KG_TO_CHUNK_PATH, 'r', encoding='utf-8') as f:
            kg_data = json.load(f)
        
        self.kg_to_chunks: Dict[str, List[str]] = {}
        for category, items in kg_data.items():
            for item in items:
                kg_id = item.get("id")
                chunk_id = item.get("chunk_id")
                if kg_id and chunk_id:
                    if isinstance(chunk_id, str):
                        self.kg_to_chunks[kg_id] = [chunk_id]
                    else:
                        self.kg_to_chunks[kg_id] = chunk_id
        print(f"  KG->Chunk 映射: {len(self.kg_to_chunks)} 条")
        
        # 加载测试数据
        with open(BENCHMARK_PATH, 'r', encoding='utf-8') as f:
            self.benchmark = json.load(f)
        print(f"  测试问题: {len(self.benchmark)} 条")
        
        # 调试：打印样例映射
        sample_kg = self.kg_ids[0] if self.kg_ids else None
        if sample_kg:
            sample_chunks = self.kg_to_chunks.get(sample_kg, ["NOT FOUND"])
            print(f"  样例映射: {sample_kg} -> {sample_chunks}")
    
    def get_chunk_ids(self, kg_id: str) -> List[str]:
        """根据 kg_id 获取对应的 chunk_id 列表"""
        return self.kg_to_chunks.get(kg_id, [kg_id])


# ============ 检索结果数据结构 ============
@dataclass
class RetrievalItem:
    """单个检索结果项"""
    kg_id: str
    chunk_ids: List[str]
    content: str
    score: float
    is_hit: bool = False  # 是否命中 GT


@dataclass
class QueryResult:
    """单个问题的检索结果"""
    query_id: int
    question: str
    gt_chunk_ids: List[str]
    retrieved_items: List[RetrievalItem]
    recall_at_k: Dict[int, float]
    hit_at_k: Dict[int, int]


@dataclass
class EvalResult:
    """评估结果"""
    model_name: str
    recall_at_k: Dict[int, float]
    hit_at_k: Dict[int, float]
    num_questions: int
    query_results: List[QueryResult]


# ============ 评估函数 ============
def calculate_metrics_at_k(retrieved_items: List[RetrievalItem], gt_chunk_ids: Set[str], k_values: List[int]) -> Tuple[Dict[int, float], Dict[int, int]]:
    """
    计算 Recall@K 和 Hit@K
    
    对于每个 K，我们看前 K 个检索结果中的所有 chunk_ids 是否覆盖了 gt_chunk_ids
    """
    recall_at_k = {}
    hit_at_k = {}
    
    for k in k_values:
        # 收集前 K 个结果的所有 chunk_ids（去重）
        retrieved_chunk_ids = set()
        for item in retrieved_items[:k]:
            for cid in item.chunk_ids:
                retrieved_chunk_ids.add(cid)
        
        # 计算命中数
        hits = len(retrieved_chunk_ids & gt_chunk_ids)
        
        # Recall@K
        if len(gt_chunk_ids) > 0:
            recall_at_k[k] = hits / len(gt_chunk_ids)
        else:
            recall_at_k[k] = 0.0
        
        # Hit@K (只要有一个命中就算 1)
        hit_at_k[k] = 1 if hits > 0 else 0
    
    return recall_at_k, hit_at_k


def mark_hits(retrieved_items: List[RetrievalItem], gt_chunk_ids: Set[str]) -> None:
    """标记每个检索结果是否命中 GT"""
    for item in retrieved_items:
        item.is_hit = bool(set(item.chunk_ids) & gt_chunk_ids)


def evaluate_embedding_only(
    embedder: BGEEmbedding, 
    data_loader: DataLoader,
    topk: int = 50
) -> Tuple[Dict[int, float], Dict[int, float], List[QueryResult]]:
    """评估仅使用 Embedding 的检索效果"""
    
    k_values = [1, 3, 5, 10, 20]
    all_recall = {k: [] for k in k_values}
    all_hit = {k: [] for k in k_values}
    query_results: List[QueryResult] = []
    
    # 构建向量索引
    print("  构建向量索引...")
    doc_embeddings = embedder.encode(data_loader.contents, show_progress=True)
    
    # 过滤有效问题
    valid_items = [
        item for item in data_loader.benchmark 
        if item.get("question") and item.get("reference_doc_id")
    ]
    print(f"  有效问题数: {len(valid_items)}")
    
    for item in tqdm(valid_items, desc="  评估问题", leave=False):
        query_id = item.get("id", 0)
        question = item["question"]
        gt_chunk_ids = set(item["reference_doc_id"])
        
        # Embedding 检索
        query_emb = embedder.encode([question])[0]
        scores = np.dot(doc_embeddings, query_emb)
        top_indices = np.argsort(scores)[::-1][:topk]
        
        # 构建检索结果项
        retrieved_items: List[RetrievalItem] = []
        for idx in top_indices:
            kg_id = data_loader.kg_ids[idx]
            chunk_ids = data_loader.get_chunk_ids(kg_id)
            retrieved_items.append(RetrievalItem(
                kg_id=kg_id,
                chunk_ids=chunk_ids,
                content=data_loader.contents[idx],
                score=float(scores[idx])
            ))
        
        # 标记命中
        mark_hits(retrieved_items, gt_chunk_ids)
        
        # 计算指标
        recall_at_k, hit_at_k = calculate_metrics_at_k(retrieved_items, gt_chunk_ids, k_values)
        
        for k in k_values:
            all_recall[k].append(recall_at_k[k])
            all_hit[k].append(hit_at_k[k])
        
        query_results.append(QueryResult(
            query_id=query_id,
            question=question,
            gt_chunk_ids=list(gt_chunk_ids),
            retrieved_items=retrieved_items[:SAVE_TOPK],
            recall_at_k=recall_at_k,
            hit_at_k=hit_at_k
        ))
    
    avg_recall = {k: np.mean(v) for k, v in all_recall.items()}
    avg_hit = {k: np.mean(v) for k, v in all_hit.items()}
    
    return avg_recall, avg_hit, query_results


def evaluate_embedding_with_reranker(
    embedder: BGEEmbedding,
    reranker: BGEReranker,
    data_loader: DataLoader,
    topk_embed: int = 50
) -> Tuple[Dict[int, float], Dict[int, float], List[QueryResult]]:
    """评估 Embedding + Reranker 的检索效果"""
    
    k_values = [1, 3, 5, 10, 20]
    all_recall = {k: [] for k in k_values}
    all_hit = {k: [] for k in k_values}
    query_results: List[QueryResult] = []
    
    # 构建向量索引
    print("  构建向量索引...")
    doc_embeddings = embedder.encode(data_loader.contents, show_progress=True)
    
    # 过滤有效问题
    valid_items = [
        item for item in data_loader.benchmark 
        if item.get("question") and item.get("reference_doc_id")
    ]
    print(f"  有效问题数: {len(valid_items)}")
    
    for item in tqdm(valid_items, desc="  评估问题", leave=False):
        query_id = item.get("id", 0)
        question = item["question"]
        gt_chunk_ids = set(item["reference_doc_id"])
        
        # Embedding 召回
        query_emb = embedder.encode([question])[0]
        scores = np.dot(doc_embeddings, query_emb)
        top_indices = np.argsort(scores)[::-1][:topk_embed]
        
        # 构建候选文档列表
        candidates = []
        for idx in top_indices:
            kg_id = data_loader.kg_ids[idx]
            chunk_ids = data_loader.get_chunk_ids(kg_id)
            candidates.append({
                "kg_id": kg_id,
                "chunk_ids": chunk_ids,
                "content": data_loader.contents[idx],
                "embed_score": float(scores[idx])
            })
        
        # Rerank
        reranked = reranker.rerank(question, candidates)
        
        # 构建检索结果项
        retrieved_items: List[RetrievalItem] = []
        for doc in reranked:
            retrieved_items.append(RetrievalItem(
                kg_id=doc["kg_id"],
                chunk_ids=doc["chunk_ids"],
                content=doc["content"],
                score=doc["rerank_score"]
            ))
        
        # 标记命中
        mark_hits(retrieved_items, gt_chunk_ids)
        
        # 计算指标
        recall_at_k, hit_at_k = calculate_metrics_at_k(retrieved_items, gt_chunk_ids, k_values)
        
        for k in k_values:
            all_recall[k].append(recall_at_k[k])
            all_hit[k].append(hit_at_k[k])
        
        query_results.append(QueryResult(
            query_id=query_id,
            question=question,
            gt_chunk_ids=list(gt_chunk_ids),
            retrieved_items=retrieved_items[:SAVE_TOPK],
            recall_at_k=recall_at_k,
            hit_at_k=hit_at_k
        ))
    
    avg_recall = {k: np.mean(v) for k, v in all_recall.items()}
    avg_hit = {k: np.mean(v) for k, v in all_hit.items()}
    
    return avg_recall, avg_hit, query_results


# ============ 保存检索结果 ============
def save_retrieval_results(result: EvalResult, output_dir: str, timestamp: str):
    """保存单个模型组合的检索结果"""
    
    # 生成安全的文件名
    safe_name = result.model_name.replace(" ", "_").replace("+", "_").replace("(", "").replace(")", "")
    txt_path = os.path.join(output_dir, f"retrieval_results_{safe_name}_{timestamp}.txt")
    
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write(f"模型: {result.model_name}\n")
        f.write(f"评估时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"问题数量: {result.num_questions}\n")
        f.write("=" * 100 + "\n\n")
        
        # 汇总指标
        f.write("【汇总指标】\n")
        f.write("-" * 50 + "\n")
        f.write(f"{'Metric':<15}")
        for k in [1, 3, 5, 10, 20]:
            f.write(f"{'Top-' + str(k):>12}")
        f.write("\n")
        f.write(f"{'Recall':<15}")
        for k in [1, 3, 5, 10, 20]:
            f.write(f"{result.recall_at_k[k]:>12.4f}")
        f.write("\n")
        f.write(f"{'Hit':<15}")
        for k in [1, 3, 5, 10, 20]:
            f.write(f"{result.hit_at_k[k]:>12.4f}")
        f.write("\n")
        f.write("-" * 50 + "\n\n")
        
        # 每个问题的详细结果
        for qr in result.query_results:
            f.write("=" * 100 + "\n")
            f.write(f"ID: {qr.query_id}\n")
            f.write(f"Question: {qr.question}\n")
            f.write("-" * 100 + "\n")
            f.write(f"Ground Truth Chunk IDs: {qr.gt_chunk_ids}\n")
            f.write(f"Recall@1: {qr.recall_at_k[1]:.4f} | Recall@5: {qr.recall_at_k[5]:.4f} | Hit@5: {qr.hit_at_k[5]}\n")
            f.write("-" * 100 + "\n")
            f.write(f"检索结果数量: {len(qr.retrieved_items)}\n")
            
            kg_ids = [item.kg_id for item in qr.retrieved_items]
            f.write(f"结果ID列表: {kg_ids}\n")
            f.write("-" * 100 + "\n")
            
            for rank, item in enumerate(qr.retrieved_items, 1):
                hit_marker = " ★HIT★" if item.is_hit else ""
                
                f.write(f"[{rank}] ID: {item.kg_id} | Score: {item.score:.4f}{hit_marker}\n")
                f.write(f"    Chunk IDs: {item.chunk_ids}\n")
                # 截断过长的内容
                content_display = item.content[:200] + "..." if len(item.content) > 200 else item.content
                f.write(f"    Content: {content_display}\n")
                f.write("\n")
            
            f.write("\n")
    
    print(f"    检索结果保存到: {txt_path}")
    return txt_path


# ============ Leaderboard 生成 ============
def run_leaderboard():
    """运行完整的 Leaderboard 评估"""
    
    print("\n" + "=" * 80)
    print("Retrieval Leaderboard 评估")
    print("=" * 80)
    
    # 加载数据
    data_loader = DataLoader()
    
    results: List[EvalResult] = []
    valid_items = [
        item for item in data_loader.benchmark 
        if item.get("question") and item.get("reference_doc_id")
    ]
    num_questions = len(valid_items)
    
    # 时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 计算总评估任务数
    num_embed_models = len(MODELS_CONFIG["embedding"])
    num_reranker_models = len(MODELS_CONFIG["reranker"])
    total_tasks = num_embed_models + num_embed_models * num_reranker_models
    
    current_task = 0
    
    # ============ 评估仅 Embedding ============
    print("\n" + "-" * 80)
    print("评估 Embedding Only 模型")
    print("-" * 80)
    
    for emb_name, emb_path in MODELS_CONFIG["embedding"].items():
        current_task += 1
        model_name = f"{emb_name} (embedding only)"
        print(f"\n[{current_task}/{total_tasks}] {model_name}")
        
        embedder = BGEEmbedding(emb_path)
        recall, hit, query_results = evaluate_embedding_only(embedder, data_loader)
        
        eval_result = EvalResult(
            model_name=model_name,
            recall_at_k=recall,
            hit_at_k=hit,
            num_questions=num_questions,
            query_results=query_results
        )
        results.append(eval_result)
        
        # 保存检索结果
        save_retrieval_results(eval_result, OUTPUT_DIR, timestamp)
        
        embedder.release()
        print(f"  ✓ 完成: Recall@5={recall[5]:.4f}, Hit@5={hit[5]:.4f}")
    
    # ============ 评估 Embedding + Reranker ============
    print("\n" + "-" * 80)
    print("评估 Embedding + Reranker 模型组合")
    print("-" * 80)
    
    for emb_name, emb_path in MODELS_CONFIG["embedding"].items():
        for rerank_name, rerank_config in MODELS_CONFIG["reranker"].items():
            current_task += 1
            model_name = f"{emb_name} + {rerank_name}"
            print(f"\n[{current_task}/{total_tasks}] {model_name}")
            
            embedder = BGEEmbedding(emb_path)
            reranker = BGEReranker(
                base_model_path=rerank_config["base"],
                adapter_path=rerank_config["adapter"]
            )
            
            recall, hit, query_results = evaluate_embedding_with_reranker(embedder, reranker, data_loader)
            
            eval_result = EvalResult(
                model_name=model_name,
                recall_at_k=recall,
                hit_at_k=hit,
                num_questions=num_questions,
                query_results=query_results
            )
            results.append(eval_result)
            
            # 保存检索结果
            save_retrieval_results(eval_result, OUTPUT_DIR, timestamp)
            
            embedder.release()
            reranker.release()
            print(f"  ✓ 完成: Recall@5={recall[5]:.4f}, Hit@5={hit[5]:.4f}")
    
    return results, num_questions, timestamp


def print_leaderboard(results: List[EvalResult], metric: str = "recall"):
    """打印 Leaderboard 表格"""
    
    k_values = [1, 3, 5, 10, 20]
    
    if metric == "recall":
        results_sorted = sorted(results, key=lambda x: x.recall_at_k[5], reverse=True)
        title = "Recall@K Leaderboard"
        get_value = lambda r, k: r.recall_at_k[k]
    else:
        results_sorted = sorted(results, key=lambda x: x.hit_at_k[5], reverse=True)
        title = "Hit@K Leaderboard"
        get_value = lambda r, k: r.hit_at_k[k]
    
    max_name_len = max(len(r.model_name) for r in results_sorted)
    name_width = max(max_name_len + 2, 50)
    
    print("\n" + "=" * (name_width + 65))
    print(title)
    print("=" * (name_width + 65))
    
    header = f"{'Rank':<6}{'Model':<{name_width}}"
    for k in k_values:
        header += f"{'Top-' + str(k):>12}"
    print(header)
    print("-" * (name_width + 65))
    
    for rank, result in enumerate(results_sorted, 1):
        row = f"{rank:<6}{result.model_name:<{name_width}}"
        for k in k_values:
            value = get_value(result, k)
            row += f"{value:>12.4f}"
        print(row)
    
    print("-" * (name_width + 65))


def save_leaderboard(results: List[EvalResult], num_questions: int, timestamp: str):
    """保存 Leaderboard 结果"""
    
    k_values = [1, 3, 5, 10, 20]
    
    # 保存 TXT
    txt_path = os.path.join(OUTPUT_DIR, f"leaderboard_{timestamp}.txt")
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("=" * 120 + "\n")
        f.write("Retrieval Leaderboard\n")
        f.write(f"评估时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"评估问题数: {num_questions}\n")
        f.write("=" * 120 + "\n\n")
        
        # Recall 表格
        f.write("【Recall@K Leaderboard】\n")
        f.write("-" * 120 + "\n")
        results_sorted = sorted(results, key=lambda x: x.recall_at_k[5], reverse=True)
        
        header = f"{'Rank':<6}{'Model':<55}"
        for k in k_values:
            header += f"{'Top-' + str(k):>12}"
        f.write(header + "\n")
        f.write("-" * 120 + "\n")
        
        for rank, result in enumerate(results_sorted, 1):
            row = f"{rank:<6}{result.model_name:<55}"
            for k in k_values:
                row += f"{result.recall_at_k[k]:>12.4f}"
            f.write(row + "\n")
        
        f.write("\n\n")
        
        # Hit 表格
        f.write("【Hit@K Leaderboard】\n")
        f.write("-" * 120 + "\n")
        results_sorted = sorted(results, key=lambda x: x.hit_at_k[5], reverse=True)
        
        f.write(header + "\n")
        f.write("-" * 120 + "\n")
        
        for rank, result in enumerate(results_sorted, 1):
            row = f"{rank:<6}{result.model_name:<55}"
            for k in k_values:
                row += f"{result.hit_at_k[k]:>12.4f}"
            f.write(row + "\n")
    
    print(f"\nLeaderboard TXT 保存到: {txt_path}")
    
    # 保存 JSON（不包含详细检索结果，避免文件过大）
    json_path = os.path.join(OUTPUT_DIR, f"leaderboard_{timestamp}.json")
    json_data = {
        "timestamp": datetime.now().isoformat(),
        "num_questions": num_questions,
        "results": [
            {
                "model_name": r.model_name,
                "recall_at_k": {str(k): v for k, v in r.recall_at_k.items()},
                "hit_at_k": {str(k): v for k, v in r.hit_at_k.items()},
            }
            for r in results
        ]
    }
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    
    print(f"Leaderboard JSON 保存到: {json_path}")
    
    return txt_path, json_path


# ============ 主函数 ============
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 运行评估
    results, num_questions, timestamp = run_leaderboard()
    
    # 打印 Leaderboard
    print_leaderboard(results, metric="recall")
    print_leaderboard(results, metric="hit")
    
    # 保存 Leaderboard
    save_leaderboard(results, num_questions, timestamp)
    
    print("\n" + "=" * 80)
    print("评估完成!")
    print(f"结果目录: {OUTPUT_DIR}")
    print("=" * 80)