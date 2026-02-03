import pandas as pd
import numpy as np
from tqdm import tqdm
import re
import ast
from collections import Counter
import os
import json
import string
import openai
import argparse

def parse_generated_answer(generated_answer):
    if isinstance(generated_answer, list):
        text = ''.join(generated_answer)
    else:
        text = generated_answer
    
    answer_matches = re.findall(r'"Answer"\s*:\s*"([^"]*)"', text)
    answer = answer_matches if answer_matches else [generated_answer]

    return answer
    
def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    return max(metric_fn(p, gt) for p in prediction for gt in ground_truths)


def calculate_acc(prediction, ground_truths):
    for gt in ground_truths:
        for p in prediction:
            if gt in p:
                return 1
    return 0


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def compute_exact(prediction, ground_truth):
    return int(normalize_answer(prediction) == normalize_answer(ground_truth))

def llm_eval_gpt4o(query, ground_truths, prediction, api_key):
    
    openai.api_key = api_key

    response = openai.chat.completions.create(
        model="gpt-4o",
        messages = [
        {"role": "system", "content": "You are an evaluation tool."},
        {"role": "user", "content": f'''Your task is to evaluate the correctness of the PREDICTED ANSWER based on the GT ANSWERs. 
                                        ### Instructions:
                                        - Read the QUERY and then compare the GT ANSWERs and the PREDICTED ANSWER.
                                        - Check if the PREDICTED ANSWER includes any of the core content of the GT ANSWERs.
                                        - If there are multiple GT ANSWERS and the PREDICTED ANSWER includes the core content of at least one of them, output "True".

                                        ### QUERY: 
                                        {query}

                                        ### GT ANSWERs: 
                                        {ground_truths}

                                        ### PREDICTED ANSWER:
                                        {prediction}

                                        ### Strictly output True or False'''}
                                        ],
        max_tokens=500,
        temperature=0.0,
    )


    # JSON 형식으로 응답 반환
    llm = response.choices[0].message.content

    
    if llm == 'True':
            llm = 1
    elif llm == 'False':
            llm = 0
    else:
        if 'True' in llm:
            print('True를 포함함')
            print(llm)
            llm = 1
        elif 'False' in llm:
            print('False를 포함함')
            print(llm)
            llm = 0
        else:
            print('llm 생성문제')
            print(llm)
            llm = 0
            
    return llm

def get_scores(prediction, ground_truths):
    
    acc = calculate_acc(prediction, ground_truths)
    em = metric_max_over_ground_truths(compute_exact, prediction, ground_truths)
    f1 = metric_max_over_ground_truths(f1_score, prediction, ground_truths)
    
    return acc, em, f1


def numbering_answers(gt_answers):
     return '\n'.join([f'({i+1}) {answer}' for i, answer in enumerate(gt_answers)])

def evaluation(args, dataset, retrieval_type):
    generation_path = f'./results/{retrieval_type}/{dataset}_generation.json'
    gt_path = f'../datasets/qrels/{dataset}.jsonl'
    
    with open(generation_path, "r", encoding='utf-8') as file:
        generation= json.load(file)
        
    gt = []
    with open(gt_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():  # 빈 줄 방지
                gt.append(json.loads(line))
    
    total_eval = {}

    total_acc, total_em, total_f1, total_llm_gpt4o= 0, 0, 0, 0
    for data in tqdm(gt, desc= f'{dataset}_{retrieval_type}'):
        q_id = data['q_id']
        ground_truth, prediction = data['answers'], parse_generated_answer(generation[q_id])
        normalized_prediction = [normalize_answer(i) for i in prediction]
        normalized_ground_truth = [normalize_answer(i) for i in ground_truth]
        
        # genertaion 평가
        acc, em, f1 = get_scores(normalized_prediction, normalized_ground_truth)
        if args.llm_eval:
            llm_gpt4o = llm_eval_gpt4o(data['query'], numbering_answers(ground_truth), '\n'.join(prediction), api_key=args.api_key)
        else:
            llm_gpt4o = 0
       
        total_eval[q_id] = {'result': {'acc': acc, 'em': em, 'f1': f1, 'llm_eval': int(llm_gpt4o)},
                            'generated_answer': prediction,
                            'gt_answers': ground_truth,
                            'n_generated_answer': normalized_prediction,
                            'n_gt_answers': normalized_ground_truth,
                            'query': data['query']}
        
        # total 성능
        total_acc += acc
        total_em += em
        total_f1 += f1
        total_llm_gpt4o += int(llm_gpt4o)
        
    
    query_nums = len(list(generation.keys()))
    a, e, ff, l1 = total_acc/query_nums, total_em/query_nums, total_f1/query_nums, total_llm_gpt4o/query_nums
    print(f'{dataset}-{retrieval_type}_generation')
    print('generation accuracy:', a)
    print('generation em:', e)
    print('generation f1:', ff)
    print('generation llm evaluation GPT4o:', l1)

    
    output_file_path = f'./results/evaluation/{retrieval_type}/{dataset}_evaluation.json'
    output_dir = os.path.dirname(output_file_path)
    os.makedirs(output_dir, exist_ok=True)
    
    with open(output_file_path, 'w', encoding='utf-8') as json_file:
        json.dump(total_eval, json_file, indent=4)

    return 
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generation Evaluation')
    parser.add_argument('--model', type=str, required=True,
                        help='Retrieval model name (e.g., aggretriever, arctic, tct_colbert, bm25, ance, splade, rerank)')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset name (e.g., msmarco, nq, lifestyle, recreation, science, technology, writing)')
    parser.add_argument('--llm_eval', type=bool, default=False,
                        help='Whether to use LLM evaluation (default: False)')
    parser.add_argument('--api_key', type=str, default='Your-OpenAI-API-Key-Here',
                        help='OpenAI API key for LLM evaluation')
    
    args = parser.parse_args()
    
    evaluation(args, dataset=args.dataset, retrieval_type=args.model)