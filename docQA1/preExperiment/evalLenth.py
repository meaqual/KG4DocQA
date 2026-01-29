import numpy as np
import json
import os
from datetime import datetime
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import matplotlib.pyplot as plt

# ========== CUDA 配置 ==========
DEVICE = "cuda:6" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# 数据文件路径
DATA_PATH = "/mnt/public/sichuan_a/hyh/queryTest1/qaSchema/xtopDoc/docQA1/preExperiment/dataSet/evalLenthPre.json"
RESULTS_DIR = "/mnt/public/sichuan_a/hyh/queryTest1/qaSchema/xtopDoc/docQA1/preExperiment/results/exp2"

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

GRANULARITY_KEYS = ["Granularity1", "Granularity2", "Granularity3", "Granularity4", "Granularity5", "Granularity6"]

class GemmaReranker:
    def __init__(self, base_path: str, adapter_path: str = None, device: str = DEVICE):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(base_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            base_path, torch_dtype=torch.float16
        ).to(self.device)
        
        if adapter_path:
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
            self.model = self.model.merge_and_unload()
        self.model.eval()
    
    def predict(self, pairs: List[List[str]], batch_size: int = 8) -> List[float]:
        scores = []
        with torch.no_grad():
            for i in range(0, len(pairs), batch_size):
                batch = pairs[i:i + batch_size]
                inputs = self.tokenizer(
                    [p[0] for p in batch], [p[1] for p in batch],
                    padding=True, truncation=True, max_length=512, return_tensors="pt"
                ).to(self.device)
                logits = self.model(**inputs).logits
                batch_scores = logits[:, -1].float().cpu().tolist() if logits.dim() == 2 else logits.squeeze(-1).float().cpu().tolist()
                if isinstance(batch_scores, (float, int)):
                    batch_scores = [batch_scores]
                scores.extend(batch_scores)
        return scores

def load_data(path: str) -> List[Dict]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_models():
    print("Loading models...")
    embedders = {name: SentenceTransformer(path, device=DEVICE) for name, path in MODELS_CONFIG["embedding"].items()}
    rerankers = {name: GemmaReranker(cfg["base"], cfg["adapter"]) for name, cfg in MODELS_CONFIG["reranker"].items()}
    return embedders, rerankers

def normalize_scores(scores: List[float]) -> List[float]:
    """归一化到 0-1 范围"""
    min_s, max_s = min(scores), max(scores)
    if max_s == min_s:
        return [0.5] * len(scores)
    return [(s - min_s) / (max_s - min_s) for s in scores]

def compute_pipeline_scores(emb_scores: List[float], rerank_scores: List[float], 
                            top_k: int = 3, alpha: float = 0.5) -> List[float]:
    """
    模拟 Embedding 召回 + Reranker 重排的 pipeline
    
    方法：
    1. 用 embedding 分数选出 top_k 候选
    2. 对这些候选用 reranker 重新打分
    3. 返回加权融合分数（对未被召回的返回很低的分数）
    """
    n = len(emb_scores)
    
    # 获取 embedding top_k 的索引
    emb_ranking = np.argsort(emb_scores)[::-1]  # 降序
    top_k_indices = set(emb_ranking[:top_k])
    
    # 融合分数：被召回的用 alpha*emb + (1-alpha)*rerank，未召回的惩罚
    pipeline_scores = []
    for i in range(n):
        if i in top_k_indices:
            # 归一化后融合
            norm_emb = (emb_scores[i] - min(emb_scores)) / (max(emb_scores) - min(emb_scores) + 1e-9)
            norm_rerank = (rerank_scores[i] - min(rerank_scores)) / (max(rerank_scores) - min(rerank_scores) + 1e-9)
            pipeline_scores.append(alpha * norm_emb + (1 - alpha) * norm_rerank)
        else:
            # 未被召回，给一个基于 embedding 的低分
            pipeline_scores.append(emb_scores[i] * 0.1)  # 惩罚
    
    return pipeline_scores

def plot_scores(query_id, query, granularities, emb_scores, rerank_scores, pipeline_scores,
                save_dir, timestamp):
    """为单个 query 绘制分数变化图（3行：Embedding / Reranker / Pipeline）"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f"Query {query_id}: {query[:80]}..." if len(query) > 80 else f"Query {query_id}: {query}", fontsize=12)
    
    x = range(len(granularities))
    x_labels = [g.replace("Granularity", "G") for g in granularities]
    
    colors_emb = ['#1f77b4', '#17becf']
    colors_rerank = ['#d62728', '#ff7f0e']
    colors_pipeline = ['#2ca02c', '#98df8a', '#9467bd', '#c5b0d5']  # 4种组合
    
    # ===== 左上：Embedding 原始分数 =====
    ax1 = axes[0, 0]
    for i, (name, scores) in enumerate(emb_scores.items()):
        ax1.plot(x, scores, marker='o', linestyle='-', color=colors_emb[i], label=name, linewidth=2)
    ax1.set_xlabel("Granularity")
    ax1.set_ylabel("Embedding Score")
    ax1.set_xticks(x)
    ax1.set_xticklabels(x_labels)
    ax1.set_title("Embedding Models")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best', fontsize=9)
    
    # ===== 右上：Reranker 原始分数 =====
    ax2 = axes[0, 1]
    for i, (name, scores) in enumerate(rerank_scores.items()):
        ax2.plot(x, scores, marker='s', linestyle='--', color=colors_rerank[i], label=name, linewidth=2)
    ax2.set_xlabel("Granularity")
    ax2.set_ylabel("Reranker Score")
    ax2.set_xticks(x)
    ax2.set_xticklabels(x_labels)
    ax2.set_title("Reranker Models")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best', fontsize=9)
    
    # ===== 左下：Pipeline 组合分数 =====
    ax3 = axes[1, 0]
    for i, (name, scores) in enumerate(pipeline_scores.items()):
        ax3.plot(x, scores, marker='^', linestyle='-', color=colors_pipeline[i], label=name, linewidth=2)
    ax3.set_xlabel("Granularity")
    ax3.set_ylabel("Pipeline Score")
    ax3.set_xticks(x)
    ax3.set_xticklabels(x_labels)
    ax3.set_title("Embedding + Reranker Pipeline (2×2 combinations)")
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='best', fontsize=8)
    
    # ===== 右下：归一化比较 =====
    ax4 = axes[1, 1]
    
    # Embedding (归一化)
    for i, (name, scores) in enumerate(emb_scores.items()):
        norm_scores = normalize_scores(scores)
        ax4.plot(x, norm_scores, marker='o', linestyle='-', color=colors_emb[i], 
                 label=f"[Emb] {name}", linewidth=1.5, alpha=0.7)
    
    # Reranker (归一化)
    for i, (name, scores) in enumerate(rerank_scores.items()):
        norm_scores = normalize_scores(scores)
        ax4.plot(x, norm_scores, marker='s', linestyle='--', color=colors_rerank[i], 
                 label=f"[Rerank] {name}", linewidth=1.5, alpha=0.7)
    
    # Pipeline (归一化)
    for i, (name, scores) in enumerate(pipeline_scores.items()):
        norm_scores = normalize_scores(scores)
        ax4.plot(x, norm_scores, marker='^', linestyle=':', color=colors_pipeline[i], 
                 label=f"[Pipe] {name}", linewidth=2)
    
    ax4.set_xlabel("Granularity")
    ax4.set_ylabel("Normalized Score (0-1)")
    ax4.set_xticks(x)
    ax4.set_xticklabels(x_labels)
    ax4.set_title("All Models Normalized Comparison")
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc='best', fontsize=7, ncol=2)
    ax4.set_ylim(-0.05, 1.05)
    
    plt.tight_layout()
    
    img_path = os.path.join(save_dir, f"query_{query_id}_{timestamp}.png")
    plt.savefig(img_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return img_path

def plot_summary(all_emb_scores, all_rerank_scores, all_pipeline_scores, 
                 granularities, save_dir, timestamp):
    """绘制所有 query 的平均分数变化图"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Average Scores Across All Queries", fontsize=14)
    
    x = range(len(granularities))
    x_labels = [g.replace("Granularity", "G") for g in granularities]
    
    colors_emb = ['#1f77b4', '#17becf']
    colors_rerank = ['#d62728', '#ff7f0e']
    colors_pipeline = ['#2ca02c', '#98df8a', '#9467bd', '#c5b0d5']
    
    # 计算平均分
    emb_names = list(all_emb_scores.keys())
    rerank_names = list(all_rerank_scores.keys())
    pipeline_names = list(all_pipeline_scores.keys())
    
    avg_emb = {name: np.mean(all_emb_scores[name], axis=0).tolist() for name in emb_names}
    avg_rerank = {name: np.mean(all_rerank_scores[name], axis=0).tolist() for name in rerank_names}
    avg_pipeline = {name: np.mean(all_pipeline_scores[name], axis=0).tolist() for name in pipeline_names}
    
    # ===== 左上：Embedding 平均分 =====
    ax1 = axes[0, 0]
    for i, name in enumerate(emb_names):
        ax1.plot(x, avg_emb[name], marker='o', linestyle='-', color=colors_emb[i], label=name, linewidth=2)
    ax1.set_xlabel("Granularity")
    ax1.set_ylabel("Avg Embedding Score")
    ax1.set_xticks(x)
    ax1.set_xticklabels(x_labels)
    ax1.set_title("Average Embedding Scores")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best', fontsize=9)
    
    # ===== 右上：Reranker 平均分 =====
    ax2 = axes[0, 1]
    for i, name in enumerate(rerank_names):
        ax2.plot(x, avg_rerank[name], marker='s', linestyle='--', color=colors_rerank[i], label=name, linewidth=2)
    ax2.set_xlabel("Granularity")
    ax2.set_ylabel("Avg Reranker Score")
    ax2.set_xticks(x)
    ax2.set_xticklabels(x_labels)
    ax2.set_title("Average Reranker Scores")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best', fontsize=9)
    
    # ===== 左下：Pipeline 平均分 =====
    ax3 = axes[1, 0]
    for i, name in enumerate(pipeline_names):
        ax3.plot(x, avg_pipeline[name], marker='^', linestyle='-', color=colors_pipeline[i], label=name, linewidth=2)
    ax3.set_xlabel("Granularity")
    ax3.set_ylabel("Avg Pipeline Score")
    ax3.set_xticks(x)
    ax3.set_xticklabels(x_labels)
    ax3.set_title("Average Pipeline Scores (Emb + Rerank)")
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='best', fontsize=8)
    
    # ===== 右下：归一化比较 =====
    ax4 = axes[1, 1]
    
    for i, name in enumerate(emb_names):
        norm_scores = normalize_scores(avg_emb[name])
        ax4.plot(x, norm_scores, marker='o', linestyle='-', color=colors_emb[i], 
                 label=f"[Emb] {name}", linewidth=1.5, alpha=0.7)
    
    for i, name in enumerate(rerank_names):
        norm_scores = normalize_scores(avg_rerank[name])
        ax4.plot(x, norm_scores, marker='s', linestyle='--', color=colors_rerank[i], 
                 label=f"[Rerank] {name}", linewidth=1.5, alpha=0.7)
    
    for i, name in enumerate(pipeline_names):
        norm_scores = normalize_scores(avg_pipeline[name])
        ax4.plot(x, norm_scores, marker='^', linestyle=':', color=colors_pipeline[i], 
                 label=f"[Pipe] {name}", linewidth=2)
    
    ax4.set_xlabel("Granularity")
    ax4.set_ylabel("Normalized Score (0-1)")
    ax4.set_xticks(x)
    ax4.set_xticklabels(x_labels)
    ax4.set_title("All Models Normalized Comparison")
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc='best', fontsize=7, ncol=2)
    ax4.set_ylim(-0.05, 1.05)
    
    plt.tight_layout()
    
    img_path = os.path.join(save_dir, f"summary_{timestamp}.png")
    plt.savefig(img_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return img_path

def main():
    data = load_data(DATA_PATH)
    print(f"Loaded {len(data)} queries")
    
    embedders, rerankers = load_models()
    print("Models loaded!\n")
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    img_dir = os.path.join(RESULTS_DIR, "images")
    os.makedirs(img_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    emb_names = list(embedders.keys())
    rerank_names = list(rerankers.keys())
    
    # Pipeline 组合名称
    pipeline_names = []
    for emb_name in emb_names:
        for rerank_name in rerank_names:
            # 简化名称
            emb_short = "E-base" if "finetuned" not in emb_name else "E-ft"
            rerank_short = "R-base" if "finetuned" not in rerank_name else "R-ft"
            pipeline_names.append(f"{emb_short}+{rerank_short}")
    
    # 表格内容
    emb_lines = []
    rerank_lines = []
    pipeline_lines = []
    
    emb_header = f"{'Query ID':<10} {'Granularity':<15} " + " ".join([f"{name:>25}" for name in emb_names])
    rerank_header = f"{'Query ID':<10} {'Granularity':<15} " + " ".join([f"{name:>30}" for name in rerank_names])
    pipeline_header = f"{'Query ID':<10} {'Granularity':<15} " + " ".join([f"{name:>20}" for name in pipeline_names])
    
    emb_lines.append("=" * len(emb_header))
    emb_lines.append("EMBEDDING SCORES BY GRANULARITY")
    emb_lines.append("=" * len(emb_header))
    emb_lines.append(emb_header)
    emb_lines.append("-" * len(emb_header))
    
    rerank_lines.append("=" * len(rerank_header))
    rerank_lines.append("RERANKER SCORES BY GRANULARITY")
    rerank_lines.append("=" * len(rerank_header))
    rerank_lines.append(rerank_header)
    rerank_lines.append("-" * len(rerank_header))
    
    pipeline_lines.append("=" * len(pipeline_header))
    pipeline_lines.append("PIPELINE SCORES (Embedding + Reranker) BY GRANULARITY")
    pipeline_lines.append("=" * len(pipeline_header))
    pipeline_lines.append(pipeline_header)
    pipeline_lines.append("-" * len(pipeline_header))
    
    # 收集所有分数用于汇总图
    all_emb_scores = {name: [] for name in emb_names}
    all_rerank_scores = {name: [] for name in rerank_names}
    all_pipeline_scores = {name: [] for name in pipeline_names}
    valid_granularities = None
    
    for idx, item in enumerate(data):
        query_id = item.get("id", idx + 1)
        query = item["query"]
        
        texts = {k: item[k] for k in GRANULARITY_KEYS if k in item}
        if not texts:
            continue
        
        text_list = list(texts.values())
        granularities = list(texts.keys())
        
        if valid_granularities is None:
            valid_granularities = granularities
        
        # ===== Embedding scores =====
        emb_scores = {}
        for name, model in embedders.items():
            q_emb = model.encode(query, normalize_embeddings=True)
            t_embs = model.encode(text_list, normalize_embeddings=True)
            emb_scores[name] = np.dot(t_embs, q_emb).tolist()
            all_emb_scores[name].append(emb_scores[name])
        
        # ===== Reranker scores =====
        rerank_scores = {}
        for name, model in rerankers.items():
            pairs = [[query, t] for t in text_list]
            rerank_scores[name] = model.predict(pairs)
            all_rerank_scores[name].append(rerank_scores[name])
        
        # ===== Pipeline scores (2x2 组合) =====
        pipeline_scores = {}
        pipe_idx = 0
        for emb_name in emb_names:
            for rerank_name in rerank_names:
                pipe_name = pipeline_names[pipe_idx]
                pipe_scores = compute_pipeline_scores(
                    emb_scores[emb_name], 
                    rerank_scores[rerank_name],
                    top_k=3,  # 召回前3
                    alpha=0.3  # reranker 权重更高
                )
                pipeline_scores[pipe_name] = pipe_scores
                all_pipeline_scores[pipe_name].append(pipe_scores)
                pipe_idx += 1
        
        # 写入表格
        for i, g in enumerate(granularities):
            emb_row = f"{query_id:<10} {g:<15} " + " ".join([f"{emb_scores[name][i]:>25.6f}" for name in emb_names])
            rerank_row = f"{query_id:<10} {g:<15} " + " ".join([f"{rerank_scores[name][i]:>30.6f}" for name in rerank_names])
            pipeline_row = f"{query_id:<10} {g:<15} " + " ".join([f"{pipeline_scores[name][i]:>20.6f}" for name in pipeline_names])
            emb_lines.append(emb_row)
            rerank_lines.append(rerank_row)
            pipeline_lines.append(pipeline_row)
        
        emb_lines.append("")
        rerank_lines.append("")
        pipeline_lines.append("")
        
        # 绘制单个 query 图片
        img_path = plot_scores(query_id, query, granularities, emb_scores, rerank_scores, 
                               pipeline_scores, img_dir, timestamp)
        print(f"Processed Query {query_id} -> {img_path}")
    
    # 保存表格
    emb_path = os.path.join(RESULTS_DIR, f"embedding_scores_{timestamp}.txt")
    rerank_path = os.path.join(RESULTS_DIR, f"reranker_scores_{timestamp}.txt")
    pipeline_path = os.path.join(RESULTS_DIR, f"pipeline_scores_{timestamp}.txt")
    
    with open(emb_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(emb_lines))
    
    with open(rerank_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(rerank_lines))
    
    with open(pipeline_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(pipeline_lines))
    
    # 绘制汇总图
    if valid_granularities:
        summary_img = plot_summary(all_emb_scores, all_rerank_scores, all_pipeline_scores,
                                   valid_granularities, img_dir, timestamp)
        print(f"\n✅ Summary image saved to: {summary_img}")
    
    print(f"\n✅ Embedding scores saved to: {emb_path}")
    print(f"✅ Reranker scores saved to: {rerank_path}")
    print(f"✅ Pipeline scores saved to: {pipeline_path}")
    print(f"✅ Individual query images saved to: {img_dir}")

if __name__ == "__main__":
    main()