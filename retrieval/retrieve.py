import argparse
import importlib.util
import sys
import os


def load_retrieval_module(model_name):
    """Load the retrieval module for the specified model."""
    # Try different possible paths
    possible_paths = [
        f'./models/beir/{model_name}.py',
        f'./models/lucene/{model_name}.py',
        f'./models/pyserini/{model_name}.py',
        f'./models/cross_encoder/{model_name}.py'
    ]
    
    module_path = None
    for path in possible_paths:
        if os.path.exists(path):
            module_path = path
            break
    
    if module_path is None:
        raise FileNotFoundError(
            f"Retrieval module not found. Tried paths: {possible_paths}\n"
            f"Please ensure your {model_name}.py file exists in one of these locations."
        )
    
    spec = importlib.util.spec_from_file_location(f"{model_name}", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"{model_name}"] = module
    spec.loader.exec_module(module)
    
    return module


def main():
    parser = argparse.ArgumentParser(description='Retrieve documents using dense retrieval models')
    parser.add_argument('--model', type=str, required=True,
                        help='Retrieval model name (e.g., aggretriever, arctic, tct_colbert, bm25, ance, splade, rerank)')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset name (e.g., msmarco, nq, lifestyle, recreation, science, technology, writing)')
    parser.add_argument('--k', type=int, default=10,
                        help='Number of documents to retrieve (default: 10)')
    
    args = parser.parse_args()
    
    print(f"[INFO] Loading {args.model} retrieval module...")
    try:
        module = load_retrieval_module(args.model)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
    
    print(f"[INFO] Running retrieval for dataset: {args.dataset} with k={args.k}")
    try:
        module.retrieve(dataset=args.dataset, k=args.k)
        print("[SUCCESS] Retrieval complete!")
    except Exception as e:
        print(f"[ERROR] Retrieval failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()