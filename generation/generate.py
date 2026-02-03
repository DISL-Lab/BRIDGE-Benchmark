import json
from tqdm import tqdm
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import argparse

model_name = "meta-llama/Llama-3.1-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.bfloat16,
    device_map="auto"
)

terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
special_tokens_dict = {"pad_token": "<pad>", "eos_token": "</s>"}
num_added_tokens = tokenizer.add_special_tokens(special_tokens_dict)


def generate(formatted_prompt):
    messages = [
        {"role": "system", "content": "You are a helpful assistant.Your task is to extract relevant information from provided documents and to answer to questions as briefly as possible."},
        {"role": "user", "content": formatted_prompt}
    ]
    
    # Get formatted text first
    formatted_text = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False
    )
    
    # Then tokenize
    inputs = tokenizer(
        formatted_text,
        return_tensors="pt"
    ).to(model.device)
    
    output_ids = model.generate(
        **inputs,
        max_new_tokens=1024,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=terminators,
    )
    
    response = output_ids[0][inputs['input_ids'].shape[-1]:]
    return tokenizer.decode(response, skip_special_tokens=True)


def format_prompt(query, retrieved_documents):
    PROMPT = f"""
Answer the given QUERY only using the information provided in the Multiple CONTEXTs. 
Do not include any assumptions, general knowledge, or information not found in the Multiple CONTEXTs.

QUERY: {query}  
Multiple CONTEXTs:  
{retrieved_documents}  


Do not provide any explanation or additional text.
Respond strictly in the following JSON format:  

- If **no relevant information** is found in CONTEXTs:  
  {{"Answer": "No relevant information found."}}

- If **relevant information** exists in CONTEXTs, answer in short form:  
  {{"Answer": "your answer"}}
"""
    return PROMPT



def main():
    parser = argparse.ArgumentParser(description='Generate answers using retrieved documents')
    parser.add_argument('--model', type=str, required=True,
                        help='Retrieval model name')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset name')
    parser.add_argument('--k', type=int, default=10,
                        help='Number of top documents to use (default: 10)')
    
    args = parser.parse_args()
    
    # Load files
    print(f"[INFO] Loading data for {args.dataset} with {args.model}...")
    
    retrieved_file_path = f'../retrieval/results/{args.model}/{args.dataset}_retrieved_corpus.json'
    query_file_path = f'../datasets/qrels/{args.dataset}.jsonl'
    
    with open(retrieved_file_path, 'r', encoding='utf-8') as f:
        retrieved_corpus = json.load(f)
    
    gt = []
    with open(query_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                gt.append(json.loads(line))
    
    # Generate answers
    print(f"[INFO] Generating answers for {len(gt)} queries...")
    generation = {}
    
    for data in tqdm(gt, desc=f'{args.dataset}-{args.model}'):
        q_id = data['q_id']

        context = '\n\n'.join([retrieved_corpus[q_id][doc_id]['text'] for doc_id in list(retrieved_corpus[q_id].keys())[:args.k]])
        prompt = format_prompt(data['query'], context)
        
        answer = generate(prompt)
        generation[q_id] = answer
    
    output_file = os.path.join(f'./results/{args.model}', f'{args.dataset}_generation.json')
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as json_file:
        json.dump(generation, json_file, indent=4)
    
    print(f"[SUCCESS] Generation complete! Saved to {output_file}")


if __name__ == "__main__":
    main()
