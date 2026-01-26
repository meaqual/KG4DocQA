# ========================test_rerank.py===========================

import torch
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '5'

# ============ 配置 ============
EMBED_MODEL_PATH = "/mnt/public/weights/bge-m3-finetune-v5"
RERANKER_BASE_MODEL = "/mnt/public/weights/recover_models/bge-reranker-v2-gemma"
RERANKER_ADAPTER_PATH = "/mnt/public/weights/bge-reranker-v2-gemma-v5"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============ Embedding ============
class BGEEmbedding:
    """BGE Embedding 模型封装"""
    
    def __init__(self, model_path: str = EMBED_MODEL_PATH):
        from transformers import AutoTokenizer, AutoModel
        import numpy as np
        
        print(f"加载 Embedding: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path).to(DEVICE)
        self.model.eval()
        print("Embedding 加载完成")
    
    def encode(self, texts, batch_size: int = 32):
        """编码文本为向量"""
        import numpy as np
        
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
    
    def compute_similarity(self, query: str, documents: list) -> list:
        """计算 query 与多个文档的相似度"""
        import numpy as np
        
        query_emb = self.encode([query])[0]
        doc_embs = self.encode(documents)
        
        # 点积 = 余弦相似度（因为已归一化）
        similarities = np.dot(doc_embs, query_emb)
        
        return similarities.tolist()


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
    
    def compute_scores(self, query: str, documents: list) -> list:
        """计算 query 与多个文档的 rerank 分数"""
        pairs = [[query, doc] for doc in documents]
        
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
        
        return scores


# ============ 测试函数 ============
def test_documents(query: str, documents: list, doc_names: list = None):
    """
    测试 query 与多个文档的 embedding 相似度和 rerank 分数
    
    Args:
        query: 查询文本
        documents: 文档列表
        doc_names: 文档名称列表（可选，用于显示）
    """
    if doc_names is None:
        doc_names = [f"文档{i+1}" for i in range(len(documents))]
    
    print("\n" + "=" * 80)
    print("测试配置")
    print("=" * 80)
    print(f"Query: {query}")
    print(f"文档数量: {len(documents)}")
    
    # 加载模型
    print("\n" + "-" * 80)
    print("加载模型...")
    print("-" * 80)
    embedder = BGEEmbedding()
    reranker = BGEReranker()
    
    # 计算分数
    print("\n" + "-" * 80)
    print("计算分数...")
    print("-" * 80)
    embed_scores = embedder.compute_similarity(query, documents)
    rerank_scores = reranker.compute_scores(query, documents)
    
    # 输出结果
    print("\n" + "=" * 80)
    print("测试结果")
    print("=" * 80)
    print(f"\nQuery: {query}\n")
    
    results = []
    for i, (name, doc, e_score, r_score) in enumerate(zip(doc_names, documents, embed_scores, rerank_scores)):
        results.append({
            "name": name,
            "doc": doc,
            "embed_score": e_score,
            "rerank_score": r_score
        })
    
    # 按 rerank 分数排序
    results.sort(key=lambda x: x["rerank_score"], reverse=True)
    
    print("-" * 80)
    print(f"{'排名':<4} {'名称':<15} {'Embed分数':<12} {'Rerank分数':<12}")
    print("-" * 80)
    
    for rank, r in enumerate(results, 1):
        print(f"{rank:<4} {r['name']:<15} {r['embed_score']:<12.4f} {r['rerank_score']:<12.4f}")
    
    print("-" * 80)
    
    # 详细内容
    print("\n" + "=" * 80)
    print("文档详情（按 Rerank 分数排序）")
    print("=" * 80)
    
    for rank, r in enumerate(results, 1):
        print(f"\n[{rank}] {r['name']}")
        print(f"    Embedding 分数: {r['embed_score']:.4f}")
        print(f"    Rerank 分数:    {r['rerank_score']:.4f}")
        print(f"    内容预览: {r['doc'][:200]}..." if len(r['doc']) > 200 else f"    内容: {r['doc']}")
    
    return results


# ============ 主函数 ============
if __name__ == "__main__":
    # ========================================
    # 在这里修改你的测试数据
    # ========================================
    
    # 测试查询
    query = "xtop的license使用策略是什么？"
    
    documents = [
        """在XTop工具中进行timing fix check时，使用的license的数量基于max_thread_number和scenario两者的最大值N的对数：licenses = log2(N)   \n特殊地，leakage优化还会需要额外check一个license，所以：\n2个scenario的leakage优化需要1+1个license；\n8个scenario的leakage优化需要3+1个license。""",
        
        """需要查看XTop许可证的使用情况，特别是浮动许可证的占用情况时""",
        
        """任务目标：检查EDA工具的license使用情况。涉及的步骤：1. 使用自带的lmutil工具，路径示例：icexplorer-xtop-2019.12.formal-Linux-x86_64-20200426/license/ 目录下的lmutil。2. 运行命令检查license server版本：lmutil lmstat。3. 运行命令检查license占用情况：lmutil lmstat -a -c port@licenseserver。注意事项：文档中未明确提到注意事项。""",
        
        """端口号和许可证服务器的主机名""",
        
        """使用XTop自带的lmutil工具检查license的使用情况。具体步骤包括定位到license目录下的lmutil脚本，运行lmutil lmstat命令查看license server版本，以及使用lmutil lmstat -a -c port@licenseserver命令检查license的占用情况。"""
    ]
    
    # 文档名称（可选）
    doc_names = []
    
    # ========================================
    # 运行测试
    # ========================================
    results = test_documents(query, documents)
    
    # 分析
    print("\n" + "=" * 80)
    print("分析")
    print("=" * 80)
    
    # 找出两种评分差异最大的文档
    max_diff = 0
    max_diff_doc = None
    for r in results:
        diff = abs(r['embed_score'] - r['rerank_score'])
        if diff > max_diff:
            max_diff = diff
            max_diff_doc = r
    
    print(f"\nEmbedding vs Rerank 差异最大的文档: {max_diff_doc['name']}")
    print(f"  Embedding: {max_diff_doc['embed_score']:.4f}")
    print(f"  Rerank:    {max_diff_doc['rerank_score']:.4f}")
    print(f"  差异:      {max_diff:.4f}")
    
    # 阈值判断
    print("\n阈值判断 (score_thresh=0.03):")
    for r in results:
        status = "✓ 通过" if r['rerank_score'] >= 0.003 else "✗ 过滤"
        print(f"  {r['name']}: {status} (rerank={r['rerank_score']:.4f})")