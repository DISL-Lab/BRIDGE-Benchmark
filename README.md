# 🌉 BRIDGE: Mitigating Evaluation Bias in IR via Multi-Agent Debate

[![ICLR 2026](https://img.shields.io/badge/ICLR-2026-blue.svg)](https://openreview.net/forum?id=DD5RNCHuzq&noteId=DD5RNCHuzq)
[![License](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-brightgreen.svg)](https://www.python.org/)
[![Dataset](https://img.shields.io/badge/HuggingFace-BRIDGE-yellow.svg)](https://huggingface.co/datasets/YOUR_HF_ID/BRIDGE)

> **Completing Missing Annotation: Multi-Agent Debate for Accurate and Scalable Relevance Assessment for IR Benchmarks**
> *Anonymous Authors* (Accepted at **ICLR 2026**)

**BRIDGE** is a refined Information Retrieval (IR) benchmark dataset designed to mitigate the "holes" problem (missing relevance labels) in existing benchmarks. By leveraging **DREAM** (Debate-based RElevance Assessment with Multi-agents), we uncovered **29,824 missing relevant chunks** across widely used datasets, enabling fairer comparison of diverse retrieval systems and more reliable RAG evaluation.

<p align="center">
  <img src="assets/figure1_overview.png" width="800" alt="DREAM Pipeline Overview">
  <br>
  <em>Figure 1: Overview of the DREAM pipeline for constructing the BRIDGE benchmark.</em>
</p>

---

## 🚀 Why BRIDGE?

Information Retrieval (IR) evaluation remains challenging due to incomplete benchmark datasets containing unlabeled relevant chunks ("holes"). This creates significant **evaluation bias**, often penalizing modern retrievers and misaligning RAG performance.

**BRIDGE** addresses this by:
1.  **Filling the Holes:** Identified **428% more gold chunks** compared to original annotations (from 6,976 to 36,800).
2.  **High-Quality Labeling:** Utilizing the **DREAM** framework, we achieved **95.2% accuracy** with only **3.5% human intervention**.
3.  **Fairer Evaluation:** Mitigates bias against advanced retrievers (e.g., Dense, Query Rewriting) that were previously underestimated.
4.  **RAG Alignment:** Introduces `RAGAlign` metric, demonstrating a stronger correlation between retrieval scores and downstream generation quality.

---

## 📚 Dataset Statistics

BRIDGE refines **7 major IR benchmark subsets** from **BEIR** and **RobustQA**, covering diverse domains including Web Search, Science, and Lifestyle.

| Dataset | Source | Domain | # Queries | Original Gold Chunks | **BRIDGE Gold Chunks** |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **MS MARCO** | BEIR | Web Search | 550 | *Legacy* | **Updated** |
| **NQ** | BEIR | Wikipedia | 550 | *Legacy* | **Updated** |
| **Lifestyle** | RobustQA | Lifestyle | 550 | *Legacy* | **Updated** |
| **Recreation** | RobustQA | Recreation | 550 | *Legacy* | **Updated** |
| **Science** | RobustQA | Science | 357 | *Legacy* | **Updated** |
| **Technology** | RobustQA | Technology | 550 | *Legacy* | **Updated** |
| **Writing** | RobustQA | Writing | 550 | *Legacy* | **Updated** |
| **Total** | - | - | **3,657** | **6,976** | **36,800** |

---

## 🏃 Quick Start
1. Loading the Dataset
Load the BRIDGE benchmark data directly from HuggingFace or using our loader script.

2. Evaluating Your Retriever
We provide a unified evaluation script compatible with standard TREC-style formatting.
