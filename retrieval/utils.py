import json
import os
from datasets import load_dataset

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        df = json.load(f)
    
    return df

def save_json(data, output_path):
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)
    return


def save_jsonl(data, output_path):
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as json_file:
        for item in data:
            json_file.write(json.dumps(item) + '\n')
    return

def load_jsonl(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():  # 빈 줄 방지
                data.append(json.loads(line))
    return data

def load_bridge_dataset(dataset_name: str) -> str:
    """Match dataset name to standard format."""
    dataset_map = {
        "msmarco": "MSMARCO",
        "nq": "NQ",
        "lifestyle": "Lifestyle",
        "recreation": "Recreation",
        "science": "Science",
        "technology": "Technology",
        "writing": "Writing"
    }
    
    bridge = load_dataset("DISLab/BRIDGE", dataset_map[dataset_name.lower()])
    df = bridge["test"]
    return df


def get_corpus(corpus_path):
    output_dic = {}
    with open(corpus_path, 'r', encoding='utf-8') as corpus_reader:
        for line in corpus_reader:
            corpus_dic = json.loads(line)
            doc_id, doc_title, doc_text = corpus_dic['_id'], corpus_dic['title'], corpus_dic['text']
            output_dic[str(doc_id)] = {'text':doc_text, 'title':doc_title}
    return output_dic

def get_query(query_path):
    output_dic = {}
    with open(query_path, 'r', encoding='utf-8') as query_reader:
        for line in query_reader:
            query_dic = json.loads(line)
            query_id, query_text = query_dic['_id'], query_dic['text']
            output_dic[str(query_id)] = query_text
    return output_dic