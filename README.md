<div align="center">
  
# 🌉 BRIDGE
## A Reliable Information Retrieval Benchmark with Complete Annotations (ICLR 2026)

[![Paper](https://img.shields.io/badge/Paper-ICLR%202026-blue)](https://openreview.net/forum?id=DD5RNCHuzq&noteId=HEIdjduzOv)
[![Dataset](https://img.shields.io/badge/Dataset-HuggingFace-yellow)](https://huggingface.co/datasets/DISLab/BRIDGE-NQ)
[![License](https://img.shields.io/badge/License-CC%20BY%204.0-green)](https://creativecommons.org/licenses/by/4.0/)

<img width="1428" height="473" alt="Image" src="https://github.com/user-attachments/assets/5dfddc1e-7a22-460e-bd9d-b8a7260e7470" />

</div>


# 🔍 Overview

**BRIDGE** is a refined Information Retrieval (IR) benchmark designed to solve the critical issue of **missing relevant chunks (holes)** in existing evaluation datasets.
Existing IR evaluation datasets often suffer from incomplete annotations, leading to unfair system comparisons.

By applying our novel **DREAM** (Debate-based RElevance Assessment with Multi-agents) framework, BRIDGE provides a more complete and reliable ground truth.

### 🌟 Key Achievements
- 🎯 Identified **29,824 missing relevant chunks** across 7 benchmark subsets
- 📈 Achieved **428% increase** over the originally annotated 6,976 gold chunks
- ✅ Maintained **95.2% labeling accuracy** with only **3.5% human involvement**
- 🔧 Enabled fairer retrieval system comparisons and more aligned RAG evaluation

## 📊 Dataset Statistics

BRIDGE significantly expands the density of relevant documents per query across diverse domains.

| Dataset | Source | Domain | # Corpus | # Queries | Avg. # C/Q (Original) | Avg. # C/Q (BRIDGE) |
|---------|--------|-----------|----------------|---------------|----------|----------|
| MS MARCO | BEIR(MS MARCO) | Web Search | 8.8M | 550 | 1.05 | 16.77 | 
| NQ | BEIR(NQ) | Web Search | 2.6M | 550 | 1.20 | 7.04 |
| Lifestyle | RobustQA(LoTTE) | Cooking, Sports, Travel| 119K | 550 | 2.30 | 6.61 |
| Recreation | RobustQA(LoTTE) | Gaming, Anime, Movies | 166K | 550 | 2.30  | 4.63 |
| Science | RobustQA(LoTTE) | Math, Physics, Biology | 1.0M | 357 | 1.90 | 16.09 |
| Technology | RobustQA(LoTTE) | Apple, Android, Security | 638K | 550 | 2.20 | 10.92 |
| Writing | RobustQA(LoTTE) | English | 199K | 550 | 2.20 | 8.52 |

---

# 🚀 Quick Start

## 1. Setup Environment
```bash
# Create a new conda environment with Python 3.10
conda create -n bridge python==3.10
conda activate bridge

# Clone the repository
git clone https://github.com/DISL-Lab/BRIDGE-Benchmark.git
cd bridge-benchmark

# Install dependencies
pip install -r requirements.txt
```

## 2. Data Preparation
**BRIDGE** provides refined annotations for existing benchmark datasets. \
We utilize seven IR benchmark test subsets:
- MS MARCO and NQ from [BEIR](https://github.com/beir-cellar/beir)
- Lifestyle, Recreation, Science, Technology, and Writing from [RobustQA](https://github.com/awslabs/robustqa-acl23)

To use **BRIDGE**, you need to download the source corpora first.


### Step 1️⃣: Download Source Corpora

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

We use `corpus.jsonl` as the source corpus.


**For LoTTE datasets (Lifestyle, Recreation, Science, Technology, Writing):**

Follow the instructions from [RobustQA GitHub](https://github.com/awslabs/robustqa-acl23):

```bash
# Clone RobustQA repository
git clone https://github.com/awslabs/robustqa-acl23.git
cd robustqa-acl23/data

# Follow their instructions to download and preprocess LoTTE datasets
wget -c "https://downloads.cs.stanford.edu/nlp/data/colbert/colbertv2/lotte.tar.gz"
tar -xvzf lotte.tar.gz
```

To replicate documents.jsonl and annotations.jsonl, run:
```bash
python ../move_lotte_files.py
python code/process_raw.py --data {lifestyle|recreation|technology|science|writing} --split {test}
```
We use `documents.jsonl` as the source corpus.

### Step 2️⃣: Download BRIDGE Annotations

Download our refined relevance annotations, query IDs, and answers from huggingface:

```python
python datasets/qrels/get_data.py
```

---

# 🧪 Evaluating Retriever on BRIDGE
We provide scripts to evaluate both standard retrievers and RAG systems using the BRIDGE benchmark.

## Using Provided Baselines
We provide several retrieval systems such as `bm25`, `ance`, `splade`, `arctic`, `tct_colbert`.

```python

# Retrieve
cd retrieval
python retrieve.py --model {retriever_name} --dataset {dataset_name} --k 10

# Evaluation
python evaluation.py --model {retriever_name} --dataset {dataset_name} --k 10
```
The retrieved results are saved at `./retrieval/results/{retriever_name}/{dataset_name}_retrieved_corpus.json` \
The evaluation results are saved at `./retrieval/results/evaluation/{retriever_name}/{dataset_name}_evaluation.json`

## Evaluating Your Own System
Put your retrieved results on path `./retrieval/results/{retriever_name}/{dataset_name}_retrieved_corpus.json`, and run:
```python
python evaluation.py --model {retriever_name} --dataset {dataset_name} --k 10
```

## Evaluating RAG Systems

```python
# Generation
cd generation
python generate.py --model {retriever_name} --dataset nq --k 10

# Evaluation
python evaluation.py --model {retriever_name} --dataset {dataset_name} --llm_eval True/False --api_key {Your-OpenAI-API-Key}
```
The generation results are saved at `./generation/results/{retriever_name}/{dataset_name}_generation.json` \
The evaluation results are saved at `./generation/results/evaluation/{retriever_name}/{dataset_name}_evaluation.json`

---


# 📈 Main Results

## Retrieval Performance Improvement

After filling annotation holes with **BRIDGE**, system rankings change significantly, and retrieval performance aligns more closely with downstream generation tasks.

<p align="center">
<img width="726" alt="Image" src="https://github.com/user-attachments/assets/5bf988a6-a01f-440c-9c69-289082701dcf" />
</p>

## Retrieval-Generation Alignment

BRIDGE significantly improves the reliability of RAG evaluation:

| Metric | Before BRIDGE | After BRIDGE | Improvement |
|--------|---------------|--------------|-------------|
| RAGAlign@10 | 0.70 | 0.84 | **+0.14** |

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

This project is licensed under **CC BY 4.0** (Creative Commons Attribution 4.0 International) - see the [LICENSE](LICENSE) file for details.

### Source Dataset Licenses

BRIDGE builds upon existing datasets. Please respect their original licenses:

- **MS MARCO**: [CC BY 4.0](https://github.com/microsoft/msmarco/blob/master/LICENSE)
- **Natural Questions (NQ)**: [Apache License 2.0](https://github.com/google-research-datasets/natural-questions/blob/master/LICENSE)
- **LoTTE (RobustQA subsets)**: No specific license specified. See [RobustQA repository disclaimers](https://github.com/awslabs/robustqa-acl23)

When using BRIDGE, please ensure compliance with the original dataset licenses.

---

## 📧 Contact

For questions or issues, please:
- Open an issue on [GitHub](https://github.com/DISL-Lab/bridge-benchmark/issues)
- Contact: [minjeong.ban@kaist.ac.kr](mailto:minjeong.ban@kaist.ac.kr) / [songhwanjun@kaist.ac.kr](mailto:songhwanjun@kaist.ac.kr)

---

<div align="center">

**⭐ If you find BRIDGE useful, please star this repository! ⭐**

</div>
