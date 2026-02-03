
import os
import json
import subprocess
from utils import *
from pyserini.search.lucene import LuceneSearcher
from tqdm import tqdm
#export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
#export PATH=$JAVA_HOME/bin:$PATH

def retrieve(dataset, k=10):
    # Indexing
    INDEX_DIR = f'./index/bm25/{dataset}'
    
    if os.path.exists(os.path.join(INDEX_DIR, 'segments.gen')):
        print(f"Index already exists at {INDEX_DIR}. Skipping indexing...")
    else:
        print(f"Creating Lucene index at {INDEX_DIR}...")
        
        DATA_DIR = f'../datasets/lucene_format'
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
            'python', '-m', 'pyserini.index.lucene',
            '--collection', 'JsonCollection',
            '--input', DATA_DIR,
            '--index', INDEX_DIR,
            '--generator', 'DefaultLuceneDocumentGenerator',
            '--threads', '4',  
            '--storePositions', '--storeDocvectors', '--storeRaw'
        ]

        subprocess.run(cmd, check=True)

        print(f"✅ Lucene index created at {INDEX_DIR}")

    # Retrieval
    df = load_bridge_dataset(dataset)
    searcher = LuceneSearcher(INDEX_DIR)
    doc_name = 'contents'
        
    result = {}
    for data in tqdm(df, desc=f"{dataset}_bm25"):
            q_id = data['q_id']
            result[q_id] = {}
            query = data['query']

            hits = searcher.search(query, k)
            
            for hit in hits:
                doc = searcher.doc(hit.docid)
                doc_dic = json.loads(doc.raw())
    
                result[q_id][doc_dic['id']] = {'score': hit.score, 'text': doc_dic[doc_name]}
    

    output_path = f'./results/bm25/{dataset}_retrieved_corpus.json'
    save_json(result, output_path)
    print(f"Results saved to {output_path}")