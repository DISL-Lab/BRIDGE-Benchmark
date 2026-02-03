import pandas as pd
import numpy as np
from tqdm import tqdm
import re
import ast
from collections import Counter
import os
import json
from utils import *
import argparse


def evaluate_retriever(retrieved_docs, gt_doc_labels):
    def eval_recall(retrieved_docs, gt_doc_labels):
        hit = 0
        for doc_id in retrieved_docs:
            if doc_id in gt_doc_labels:
                hit += 1
        return hit/len(gt_doc_labels)

    def eval_precision(retrieved_docs, gt_doc_labels):
        hit = 0
        for doc_id in retrieved_docs:
            if doc_id in gt_doc_labels:
                hit += 1
        return hit/len(retrieved_docs)
    
    def eval_ndcg(retrieved_docs, gt_doc_labels):
        dcg = 0
        idcg = sum([1/np.log2(i+2) for i in range(len(gt_doc_labels))])
        for i in range(len(retrieved_docs)):
            if retrieved_docs[i] in gt_doc_labels:
                dcg += 1/np.log2(i+2)
        ndcg = dcg/idcg
        return ndcg
    
    recall, precision, ndcg = eval_recall(retrieved_docs, gt_doc_labels), eval_precision(retrieved_docs, gt_doc_labels), eval_ndcg(retrieved_docs, gt_doc_labels)
    if recall+precision == 0:
        f1 = 0
    else:
        f1 = (2*recall*precision)/(recall+precision)
    
    if recall > 0:
        hitrate = 1
    else:
        hitrate = 0
        
    return hitrate, recall, precision, f1, ndcg

def evaluation(dataset, retrieval_model, k):

    gt = load_jsonl(f'../datasets/qrels/{dataset}.jsonl')
    retrieved_corpus = load_json(f'./results/{retrieval_model}/{dataset}_retrieved_corpus.json')
    
    total_eval = {}
    total_hitrate, total_retriever_recall, total_retriever_precision, total_retriever_f1, total_ndcg = 0, 0, 0, 0, 0
    for data in gt:
        q_id = data['q_id']
        if len(data['doc_id']) > 0:
            hitrate, retriever_recall, retriever_precision, retriever_f1, ndcg = evaluate_retriever(list(retrieved_corpus[q_id].keys())[:k], data['doc_id']) 
            total_eval[q_id] = {'hitrate': hitrate, 'recall': retriever_recall, 'precision': retriever_precision, 'f1': retriever_f1, 'ndcg':ndcg }
        
            total_hitrate += hitrate
            total_retriever_recall += retriever_recall
            total_retriever_precision += retriever_precision
            total_retriever_f1 += retriever_f1
            total_ndcg += ndcg

        
    
    query_nums = len(total_eval)
    h, r1, p1, f, n= total_hitrate/query_nums, total_retriever_recall/query_nums, total_retriever_precision/query_nums, total_retriever_f1/query_nums, total_ndcg/query_nums
    print(f'{dataset}-{retrieval_model}_Top{k}')
    print('retriever hitrate:', h)
    print('retriever recall:', r1)
    print('retriever precision:', p1)
    print('retriever f1:', f)
    print('retriever ndcg:', n)

    save_json(total_eval, f'./results/evaluation/{retrieval_model}/{dataset}_evaluation.json')

    return 
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Retrieval Evaluation')
    parser.add_argument('--model', type=str, required=True,
                        help='Retrieval model name (e.g., aggretriever, arctic, tct_colbert, bm25, ance, splade, rerank)')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset name (e.g., msmarco, nq, lifestyle, recreation, science, technology, writing)')
    parser.add_argument('--k', type=int, default=10,
                        help='Number of documents to retrieve (default: 10)')
    
    args = parser.parse_args()
    
    evaluation(dataset=args.dataset, retrieval_model=args.model, k=args.k)