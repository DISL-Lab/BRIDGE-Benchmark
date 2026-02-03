from unittest import result
from pyserini.encode import ArcticQueryEncoder
from pyserini.search.faiss import FaissSearcher
import json, os
from tqdm import tqdm
from utils import *
from models.pyserini.indexer import index

# https://github.com/castorini/pyserini/blob/master/docs/usage-search.md#learned-dense-retrieval-models

def retrieve(dataset, k=10):
    df = load_bridge_dataset(dataset)
    output_path = f'./retrieval/results/arctic/{dataset}_retrieved_corpus.json'
    
    # Indexing
    INDEX_DIR = f'./retrieval/index/arctic/{dataset}'
    if not os.path.exists(INDEX_DIR):
        print("Indexing started...")
        index('arctic', dataset)
    else:
        print("Index already exists. Skipping indexing...")

    corpus = load_jsonl(f'../datasets/pyserini_format/{dataset}_corpus.jsonl')
    # Retrieval
    encoder = ArcticQueryEncoder('Snowflake/snowflake-arctic-embed-m')
    faiss_searcher = FaissSearcher(
        INDEX_DIR,
        encoder)
    
    result = {}
    for data in tqdm(df, desc=f"{dataset}_arctic"):
        q_id = data['q_id']
        result[q_id] = {}
        query = data['query']
        hits = faiss_searcher.search(query, k)

        if dataset == 'nq':
            for hit in hits:
                result[q_id][hit.docid] = {'score': f'{hit.score:.5f}', 'text': corpus[int(hit.docid[3:])]['contents']}
        else:
            for hit in hits:
                result[q_id][hit.docid] = {'score': f'{hit.score:.5f}', 'text': corpus[int(hit.docid)]['contents']}
        
    save_json(result, output_path)
    print(f"Results saved to {output_path}")