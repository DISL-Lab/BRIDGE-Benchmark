# 🌉 BRIDGE: A Reliable Information Retrieval Benchmark with Complete Annotations

<div align="center">

[![Paper](https://img.shields.io/badge/Paper-ICLR%202026-blue)](link-to-paper)
[![Dataset](https://img.shields.io/badge/Dataset-HuggingFace-yellow)](link-to-huggingface)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

</div>

---

## 📢 News

- **[2026-02]** BRIDGE dataset is now available on HuggingFace!
- **[2026-01]** Paper accepted to ICLR 2026!

---

## 🔍 Overview

BRIDGE is a refined IR benchmark dataset that addresses the critical problem of **missing relevant chunks (holes)** in existing IR evaluation datasets. By applying our novel **DREAM** (Debate-based RElevance Assessment with Multi-agents) framework, we:

- 🎯 Identified **29,824 missing relevant chunks** across 7 benchmark subsets
- 📈 Achieved **428% increase** over the originally annotated 6,976 gold chunks
- ✅ Maintained **95.2% labeling accuracy** with only **3.5% human involvement**
- 🔧 Enabled fairer retrieval system comparisons and more reliable RAG evaluation

### Key Statistics

| Dataset | Source | Domain | # Corpus | # Queries | Avg. # C/Q in Original | Avg. # C/Q in BRIDGE |
|---------|--------|-----------|----------------|---------------|----------|----------|
| MS MARCO | MS MARCO/BEIR | Web Search | 8,841,823 | 550 | 1.05 | 16.77 | 
| NQ | NQ/BEIR | Web Search | 2,681,468 | 550 | 1.20 | 7.04 |
| Lifestyle | LoTTE/RobustQA | Cooking, Sports, Travel| 119,461 | 550 | 2.30 | 6.61 |
| Recreation | LoTTE/RobustQA | Gaming, Anime, Movies | 166,975 | 550 | 2.30  | 4.63 |
| Science | LoTTE/RobustQA | Math, Physics, Biology | 1,000,000 | 357 | 1.90 | 16.09 |
| Technology | LoTTE/RobustQA | Apple, Android, Security | 638,509 | 550 | 2.20 | 10.92 |
| Writing | LoTTE/RobustQA | English | 199,994 | 550 | 2.20 | 8.52 |

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

BRIDGE provides refined annotations for existing benchmark datasets. To use BRIDGE, you need to download the source corpora and apply our annotations.

### Step 1: Download Source Corpora

**For MS MARCO & NQ:**
```bash
# Install BEIR
pip install beir

# Download MS MARCO corpus
python -c "from beir import util; util.download_and_unzip('https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/msmarco.zip', 'datasets')"

# Download NQ corpus
python -c "from beir import util; util.download_and_unzip('https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nq.zip', 'datasets')"
```

Alternatively, manually download the dataset at [BEIR GitHub](https://github.com/beir-cellar/beir):


**For LoTTE datasets (Lifestyle, Recreation, Science, Technology, Writing):**

Follow the instructions from [RobustQA GitHub](https://github.com/awslabs/robustqa-acl23):

```bash
# Clone RobustQA repository
git clone https://github.com/awslabs/robustqa-acl23.git
cd robustqa-acl23

# Follow their instructions to download and preprocess LoTTE datasets
# The LoTTE datasets include: lifestyle, recreation, science, technology, writing
```
- Download raw data here: [https://downloads.cs.stanford.edu/nlp/data/colbert/colbertv2/lotte.tar.gz](https://downloads.cs.stanford.edu/nlp/data/colbert/colbertv2/lotte.tar.gz) into robustqa-acl23/data/lotte.
- Annotations: there is no data license specfied [https://github.com/stanford-futuredata/ColBERT/blob/main/LoTTE.md.](https://github.com/stanford-futuredata/ColBERT/blob/main/LoTTE.md.) We only keep doc_id and qid in the published annotation files.
- To replicate documents.jsonl and annotations.jsonl, run:
```bash
python code/process_raw.py --dataset {lifestyle|recreation|technology|science|writing} --split {test|dev}
```
Please refer to the RobustQA repository for detailed preprocessing steps.

### Step 2: Download BRIDGE Annotations

We provide the refined relevance annotations, query IDs, relevant document IDs, and answers for all datasets:

**Option A: HuggingFace (Recommended)**
```python
from datasets import load_dataset

# Load BRIDGE annotations
bridge_annotations = load_dataset("your-org/bridge-benchmark")

# Access annotations for each dataset
ms_marco_bridge = bridge_annotations["ms_marco"]
nq_bridge = bridge_annotations["nq"]
lifestyle_bridge = bridge_annotations["lifestyle"]
recreation_bridge = bridge_annotations["recreation"]
science_bridge = bridge_annotations["science"]
technology_bridge = bridge_annotations["technology"]
writing_bridge = bridge_annotations["writing"]
```

**Option B: Direct Download**
```bash
# Download BRIDGE annotations
wget https://your-link/bridge_annotations.zip
unzip bridge_annotations.zip
```

### Step 3: Merge Corpus with BRIDGE Annotations
```python
from bridge import merge_annotations

# Merge source corpus with BRIDGE annotations
bridge_dataset = merge_annotations(
    corpus_path="datasets/msmarco/corpus.jsonl",  # Path to BEIR corpus
    annotations_path="bridge_annotations/ms_marco/qrels_bridge.tsv",
    queries_path="bridge_annotations/ms_marco/queries.jsonl",
    answers_path="bridge_annotations/ms_marco/answers.jsonl"
)
```

Or use our automated script:
```bash
# Merge all datasets at once
python scripts/build_bridge.py \
    --beir_path datasets/ \
    --lotte_path robustqa-acl23/data/lotte/ \
    --bridge_annotations bridge_annotations/ \
    --output_path bridge_complete/
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
