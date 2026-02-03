import os
import json
import subprocess
from utils import *
#https://github.com/castorini/pyserini/blob/master/docs/usage-search.md#learned-dense-retrieval-models

#encoder_type = 'BAAI/contriever-m3'
#encoder_name = 'BAAI/contriever-m3'

retriever_encoder = {'arctic': 'Snowflake/snowflake-arctic-embed-m',
                     'aggretriever': 'castorini/aggretriever-cocondenser',
                     'tct_colbert': 'castorini/tct_colbert-v2-hnp-msmarco'}


def index(model_name, dataset):
    INDEX_DIR = f'./index/{model_name}/{dataset}'
    
    DATA_DIR = f'../datasets/pyserini_format'
    if os.path.exists(os.path.join(DATA_DIR, f'{dataset}_corpus.jsonl')):
        pass
    else:
        os.makedirs(DATA_DIR, exist_ok=True)
        jsonl_path = os.path.join(DATA_DIR, f'{dataset}_corpus.jsonl')
        if dataset in ['msmarco', 'nq']:
            corpus = load_jsonl(f'../datasets/{dataset}/corpus.jsonl')
            with open(jsonl_path, 'w', encoding='utf-8') as f:
                for data in corpus:
                    json_obj = {
                        'id': str(data['_id']),  
                        'contents': data['text'].replace('\n', ' ')
                    }
                    f.write(json.dumps(json_obj, ensure_ascii=False) + '\n')
        else:
            corpus = load_jsonl(f'../robustqa-acl23/data/{dataset}/test/documents.jsonl')
            with open(jsonl_path, 'w', encoding='utf-8') as f:
                for data in corpus:
                    json_obj = {
                        'id': str(data['doc_id']),  
                        'contents': data['text'].replace('\n', ' ')
                    }
                    f.write(json.dumps(json_obj, ensure_ascii=False) + '\n')
    
    

    cmd = [
        'python', '-m', 'pyserini.encode',
        'input',
        '--corpus', DATA_DIR,
        '--fields', 'text',
        '--delimiter', '\n',
        '--shard-id', '0',
        '--shard-num', '1',
        'output',
        '--embeddings', INDEX_DIR,
        '--to-faiss',
        'encoder',
        '--encoder', retriever_encoder[model_name],
        '--encoder-class', model_name,
        '--fields', 'text',
        '--batch', '32',
        '--fp16'
    ]


    subprocess.run(cmd, check=True)

    print(f"✅ {model_name} index created at {INDEX_DIR}")

'''
document_encoder_class_map = {
    "dpr": DprDocumentEncoder,
    "tct_colbert": TctColBertDocumentEncoder,
    "aggretriever": AggretrieverDocumentEncoder,
    "ance": AnceDocumentEncoder,
    "sentence-transformers": AutoDocumentEncoder,
    "unicoil": UniCoilDocumentEncoder,
    "openai-api": OpenAiDocumentEncoder,
    "cosdpr": CosDprDocumentEncoder,
    "auto": AutoDocumentEncoder,
    "clip": ClipDocumentEncoder,
    "contriever": AutoDocumentEncoder,
    "arctic": ArcticDocumentEncoder,
}

query_encoder_class_map = {
    "dkrr": DkrrDprQueryEncoder,
    "cosdpr": CosDprQueryEncoder,
    "dpr": DprQueryEncoder,
    "bpr": BprQueryEncoder,
    "tct_colbert": TctColBertQueryEncoder,
    "ance": AnceQueryEncoder,
    "sentence": AutoQueryEncoder,
    "contriever": AutoQueryEncoder,
    "aggretriever": AggretrieverQueryEncoder,
    "openai-api": OpenAiQueryEncoder,
    "auto": AutoQueryEncoder,
    "clip": ClipQueryEncoder,
    "arctic": ArcticQueryEncoder,
}'''