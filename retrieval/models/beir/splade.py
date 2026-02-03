import logging
import os
import pathlib
import random

from beir.logging import LoggingHandler
from beir import util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval import models
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from utils import *


def retrieve(dataset, k=10):
    corpus_path = f'../datasets/beir_format/{dataset}_corpus.jsonl'
    query_path = f'../datasets/beir_format/{dataset}_queries.jsonl'
    
    if dataset in ['msmarco', 'nq']:
        corpus_path = f'../datasets/{dataset}/corpus.jsonl'
    else:
        if not os.path.exists(corpus_path):
            print(f"Creating beir format file for {dataset}")
            # corpus
            os.makedirs(os.path.dirname(corpus_path), exist_ok=True)
            corpus = load_jsonl(f'../robustqa-acl23/data/{dataset}/test/documents.jsonl')
            with open(corpus_path, 'w', encoding='utf-8') as f:
                for data in corpus:
                    new_data = {"_id": data['doc_id'], "title": "", "text": data['text']}
                    json_line = json.dumps(new_data, ensure_ascii=False)
                    f.write(json_line + '\n')
    # queries
    if not os.path.exists(query_path):
        df = load_bridge_dataset(dataset)
        os.makedirs(os.path.dirname(query_path), exist_ok=True)
        with open(query_path, 'w', encoding='utf-8') as f:
            for data in df:
                q_id = data['q_id']
                data = {"_id": q_id, "text": data['query']}
                json_line = json.dumps(data, ensure_ascii=False)
                f.write(json_line + '\n')
        
    corpus, queries = get_corpus(corpus_path), get_query(query_path)

    output_path = f'./results/splade/{dataset}_retrieved_corpus.json'
    #### SPARSE Retrieval using SPLADE ####
    # The SPLADE model provides a weight for each query token and document token
    # The final score is taken using a dot-product between the weights of the common tokens.
    # To learn more, please refer to the link below:
    # https://europe.naverlabs.com/blog/splade-a-sparse-bi-encoder-bert-based-model-achieves-effective-and-efficient-first-stage-ranking/

    #################################################
    #### 1. Loading SPLADE model from NAVER LABS ####
    #################################################
    # Sadly, the model weights from SPLADE are not on huggingface etc.
    # The SPLADE v1 model weights are available on their original repo: (https://github.com/naver/splade)

    # First clone SPLADE GitHub repo: git clone https://github.com/naver/splade.git
    # NOTE: this version only works for max agg in SPLADE!

    model_path = "naver/splade-cocondenser-ensembledistil"
    model = DRES(models.SPLADE(model_path), batch_size=128)
    retriever = EvaluateRetrieval(model, score_function="dot")

    #### Retrieve dense results (format of results is identical to qrels)
    results = retriever.retrieve(corpus, queries)


    filtered_results = {}

    for query_id, doc_scores in results.items():
        top_k_docs = dict(sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:k])
        filtered_results[query_id] = {}
        for doc_id in top_k_docs:
            filtered_results[query_id][doc_id] = {'score': f'{top_k_docs[doc_id]:.5f}', 'text': corpus[doc_id]['contents']}

    save_json(filtered_results, output_path)
    print(f"Results saved to {output_path}")
    