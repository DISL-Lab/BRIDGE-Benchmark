# 🌉 BRIDGE: A Reliable Information Retrieval Benchmark with Complete Annotations

<div align="center">

[![Paper](https://img.shields.io/badge/Paper-ICLR%202026-blue)](link-to-paper)
[![Dataset](https://img.shields.io/badge/Dataset-HuggingFace-yellow)](link-to-huggingface)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

**Debate-based RElevance Assessment with Multi-agents for Accurate and Scalable IR Benchmarks**

[Paper](link-to-paper) | [Dataset](link-to-dataset) | [Code](link-to-code)

</div>

---

## 📢 News

- **[2026-XX]** BRIDGE dataset is now available on HuggingFace!
- **[2026-XX]** Paper accepted to ICLR 2026!

---

## 🔍 Overview

BRIDGE is a refined IR benchmark dataset that addresses the critical problem of **missing relevant chunks (holes)** in existing IR evaluation datasets. By applying our novel **DREAM** (Debate-based RElevance Assessment with Multi-agents) framework, we:

- 🎯 Identified **29,824 missing relevant chunks** across 7 benchmark subsets
- 📈 Achieved **428% increase** over the originally annotated 6,976 gold chunks
- ✅ Maintained **95.2% labeling accuracy** with only **3.5% human involvement**
- 🔧 Enabled fairer retrieval system comparisons and more reliable RAG evaluation

### Key Statistics

| Dataset | Domain | # Queries | # Original Gold | # BRIDGE Gold | Increase |
|---------|--------|-----------|----------------|---------------|----------|
| MS MARCO | Web Search | 550 | 550 | 9,224 | +1,577% |
| NQ | Web Search | 550 | 3,874 | 7,748 | +100% |
| Lifestyle | Community QA | 550 | 550 | 3,636 | +561% |
| Recreation | Community QA | 550 | 550 | 2,547 | +363% |
| Science | Community QA | 357 | 357 | 5,744 | +1,509% |
| Technology | Community QA | 550 | 550 | 6,005 | +992% |
| Writing | Community QA | 550 | 545 | 4,896 | +798% |
| **Total** | - | **3,657** | **6,976** | **36,800** | **+428%** |

---

## 🚀 Quick Start

### Installation
```bash
# Clone the repository
git clone https://github.com/your-org/bridge-benchmark.git
cd bridge-benchmark

# Install dependencies
pip install -r requirements.txt
```

### Download BRIDGE Dataset

BRIDGE provides refined annotations for existing benchmark datasets. Due to licensing requirements, we provide different access methods for each source dataset:

#### Option 1: Direct Download (Recommended)

Download the complete BRIDGE annotations from HuggingFace:
```python
from datasets import load_dataset

# Load BRIDGE dataset
bridge_data = load_dataset("your-org/bridge-benchmark")

# Access specific subsets
ms_marco = bridge_data["ms_marco"]
nq = bridge_data["nq"]
lifestyle = bridge_data["lifestyle"]
# ... and more
```

#### Option 2: Build from Source Datasets

For full reproducibility, you can reconstruct BRIDGE by combining source datasets with our annotations:

**Step 1: Download Source Datasets**

- **MS MARCO & NQ (via BEIR)**: 
```bash
  # These are publicly available under Apache 2.0 license
  python scripts/download_beir.py --datasets msmarco nq
```

- **RobustQA (Lifestyle, Recreation, Science, Technology, Writing)**:
```bash
  # Download from official RobustQA repository
  git clone https://github.com/awslabs/robustqa-acl23.git
  # Follow their instructions to download LoTTE datasets
```

**Step 2: Apply BRIDGE Annotations**
```python
from bridge import load_bridge_annotations, merge_annotations

# Load our refined annotations
annotations = load_bridge_annotations()

# Merge with source datasets
bridge_dataset = merge_annotations(
    source_datasets=["ms_marco", "nq", "lifestyle", ...],
    annotations=annotations
)
```

---

## 📊 Evaluating Your Retriever on BRIDGE

### Basic Evaluation
```python
from bridge import BRIDGEvaluator

# Initialize evaluator
evaluator = BRIDGEvaluator(dataset="ms_marco")

# Your retriever function: query -> List[retrieved_chunks]
def your_retriever(query):
    # Your retrieval logic here
    return retrieved_chunks

# Evaluate
results = evaluator.evaluate(
    retriever=your_retriever,
    metrics=["hit@10", "ndcg@10", "recall@10"]
)

print(f"Hit@10: {results['hit@10']:.3f}")
print(f"nDCG@10: {results['ndcg@10']:.3f}")
```

### Benchmark 25 Retrieval Systems

We provide ready-to-use implementations of 25 retrieval systems evaluated in our paper:
```python
from bridge.retrievers import BM25, SPLADE, DPR, Arctic, get_all_retrievers

# Evaluate single retriever
bm25 = BM25()
results = evaluator.evaluate(bm25, metrics=["hit@10", "ndcg@10"])

# Evaluate all 25 systems
all_retrievers = get_all_retrievers()
benchmark_results = evaluator.benchmark(all_retrievers)
```

### RAG Evaluation

Evaluate retrieval-generation alignment:
```python
from bridge import RAGEvaluator

# Initialize RAG evaluator
rag_eval = RAGEvaluator(
    dataset="ms_marco",
    generator="llama3.1-8b-instruct"
)

# Evaluate RAG pipeline
rag_results = rag_eval.evaluate(
    retriever=your_retriever,
    metrics=["hit@10", "rag_align@10", "generation_accuracy"]
)

print(f"Retrieval-Generation Alignment: {rag_results['rag_align@10']:.3f}")
```

---

## 🔬 DREAM Framework

Our DREAM (Debate-based RElevance Assessment with Multi-agents) framework enables high-quality annotation with minimal human effort:

### Key Features

- **🤖 Multi-Agent Debate**: Two LLM agents with opposing stances debate chunk relevance
- **🔄 Iterative Refinement**: Multi-round reciprocal critique until consensus or escalation
- **👥 Smart Escalation**: Only 3.5% of cases escalated to humans based on disagreement
- **📚 Debate History**: Provides context to human annotators for informed decisions

### Using DREAM for Your Own Annotations
```python
from dream import DREAMAnnotator

# Initialize DREAM
annotator = DREAMAnnotator(
    model="llama3.3-70b-instruct",
    max_rounds=2,
    temperature=0.0
)

# Annotate query-chunk pairs
results = annotator.annotate(
    queries=your_queries,
    chunks=your_chunks,
    answers=your_answers
)

# Access consensus and escalated cases
consensus = results["consensus"]  # 96.5% of cases
escalated = results["escalated"]  # 3.5% for human review
```

---

## 📈 Main Results

### Retrieval Performance Improvement

After filling holes with BRIDGE, all 25 retrieval systems show improved Hit@10 scores:

| System Type | Example Systems | Avg. Hit@10 Gain |
|-------------|----------------|------------------|
| Sparse | BM25, SPLADE | +0.17 |
| Dense | DPR, Arctic, ANCE | +0.15 |
| BM25 + Rerank | TinyBERT, MonoT5 | +0.15 |
| ANCE + Rerank | TinyBERT, MonoT5 | +0.14 |
| BM25 + Rewrite | HyDE, Q2D, MuGI | **+0.25** |

### Retrieval-Generation Alignment

BRIDGE significantly improves RAG evaluation reliability:

| Metric | Before BRIDGE | After BRIDGE | Improvement |
|--------|---------------|--------------|-------------|
| RAGAlign@10 | 0.70 | 0.84 | **+0.14** |
| Pearson Correlation | 0.87 | 0.985 | +0.115 |

---

## 📖 Citation

If you use BRIDGE in your research, please cite our paper:
```bibtex
@inproceedings{bridge2026,
  title={Completing Missing Annotation: Multi-Agent Debate for Accurate and Scalable Relevance Assessment for IR Benchmarks},
  author={Anonymous Authors},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2026}
}
```

---

## 📁 Dataset Structure
```
bridge-benchmark/
├── data/
│   ├── ms_marco/
│   │   ├── queries.jsonl
│   │   ├── corpus.jsonl
│   │   ├── qrels_original.tsv
│   │   └── qrels_bridge.tsv
│   ├── nq/
│   ├── lifestyle/
│   ├── recreation/
│   ├── science/
│   ├── technology/
│   └── writing/
├── annotations/
│   ├── consensus_labels.jsonl
│   ├── human_labels.jsonl
│   └── debate_history.jsonl
└── scripts/
    ├── download_beir.py
    ├── download_robustqa.py
    └── merge_annotations.py
```

### Data Format

**qrels_bridge.tsv** (TREC format):
```
query-id  0  chunk-id  relevance
q1        0  doc1      1
q1        0  doc2      0
```

**debate_history.jsonl**:
```json
{
  "query_id": "q1",
  "chunk_id": "doc1",
  "round_1": {
    "agent_a": {"stance": "relevant", "reasoning": "...", "label": 1},
    "agent_b": {"stance": "irrelevant", "reasoning": "...", "label": 0}
  },
  "round_2": {...},
  "final_label": 1,
  "consensus": true
}
```

---

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Source Dataset Licenses

BRIDGE builds upon existing datasets with the following licenses:

- **MS MARCO**: [Microsoft Research Data License](https://microsoft.github.io/msmarco/)
- **Natural Questions (NQ)**: [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/)
- **LoTTE (RobustQA subsets)**: Please refer to [RobustQA repository](https://github.com/awslabs/robustqa-acl23)

When using BRIDGE, please ensure compliance with the original dataset licenses.

---

## 🙏 Acknowledgments

We thank the creators of BEIR and RobustQA for providing the foundational benchmarks that made this work possible.

---

## 📧 Contact

For questions or issues, please:
- Open an issue on [GitHub](https://github.com/your-org/bridge-benchmark/issues)
- Contact: [your-email@domain.com](mailto:your-email@domain.com)

---

<div align="center">

**⭐ If you find BRIDGE useful, please star this repository! ⭐**

</div>
