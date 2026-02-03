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
- 🔧 Enabled fairer retrieval system comparisons and more aligned RAG evaluation

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

BRIDGE provides refined annotations for existing benchmark datasets. We utilize seven IR benchmark test subsets: MS MARCO and NQ from [BEIR](https://github.com/beir-cellar/beir); and Lifestyle, Recreation, Science, Technology, and Writing from [RobustQA](https://github.com/awslabs/robustqa-acl23). 

To use BRIDGE, you need to download the source corpora and apply our annotations.


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

Alternatively, manually download the dataset at [BEIR GitHub](https://github.com/beir-cellar/beir). 

We adopt corpus.jsonl as a corpus.


**For LoTTE datasets (Lifestyle, Recreation, Science, Technology, Writing):**

Follow the instructions from [RobustQA GitHub](https://github.com/awslabs/robustqa-acl23):

```bash
# Clone RobustQA repository
git clone https://github.com/awslabs/robustqa-acl23.git
cd robustqa-acl23/data

# Follow their instructions to download and preprocess LoTTE datasets
# The LoTTE datasets include: lifestyle, recreation, science, technology, writing
wget -c "https://downloads.cs.stanford.edu/nlp/data/colbert/colbertv2/lotte.tar.gz"
tar -xvzf lotte.tar.gz
```
- Download raw data here: [https://downloads.cs.stanford.edu/nlp/data/colbert/colbertv2/lotte.tar.gz](https://downloads.cs.stanford.edu/nlp/data/colbert/colbertv2/lotte.tar.gz) into robustqa-acl23/data.
- Annotations: there is no data license specfied [https://github.com/stanford-futuredata/ColBERT/blob/main/LoTTE.md](https://github.com/stanford-futuredata/ColBERT/blob/main/LoTTE.md). We only keep doc_id and qid in the published annotation files.
- To replicate documents.jsonl and annotations.jsonl, run:
```bash
python ../move_lotte_files.py
python code/process_raw.py --data {lifestyle|recreation|technology|science|writing} --split {test}
```
We adopt documents.jsonl as a corpus.

### Step 2: Download BRIDGE Annotations

We provide the refined relevance annotations, query IDs, relevant document IDs, and answers for all datasets on huggingface:

```python
python datasets/qrels/get_data.py
```

---

## 📊 Evaluating Retriever on BRIDGE

### Retrieval Systems Example

We provide some retrieval systems evaluated in our paper:
```python

# Retrieve
cd retrieval
python retrieve.py --model {retriever_name} --dataset {dataset_name} --k 10

# Evaluation
python evaluation.py --model {retriever_name} --dataset {dataset_name} --k 10
```

### Evaluate Your Retrieval System
Put your retrieved results on path retrieval/results/{retriever_name}/{dataset_name}_retrieved_corpus.json, and run:
```python
python evaluation.py --model {retriever_name} --dataset {dataset_name} --k 10
```

### RAG Evaluation

```python
# Generation
cd generation
CUDA_VISIBLE_DEVICES=6 python generate.py --model tct_colbert --dataset nq --k 10

# Evaluation
python evaluation.py --model {retriever_name} --dataset {dataset_name} --llm_eval True --api_key {Your-OpenAI-API-Key}
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
