import os
from datasets import load_dataset


msmarco_bridge = load_dataset("DISLab/BRIDGE-MSMARCO")
nq_bridge = load_dataset("DISLab/BRIDGE-NQ")
lifestyle_bridge = load_dataset("DISLab/BRIDGE-LoTTE", 'Lifestyle')
recreation_bridge = load_dataset("DISLab/BRIDGE-LoTTE", 'Recreation')
science_bridge = load_dataset("DISLab/BRIDGE-LoTTE", 'Science')
technology_bridge = load_dataset("DISLab/BRIDGE-LoTTE", 'Technology')
writing_bridge = load_dataset("DISLab/BRIDGE-LoTTE", 'Writing')

SAVE_ROOT = "./datasets/qrels"
os.makedirs(SAVE_ROOT, exist_ok=True)


datasets_to_save = {
    "msmarco": msmarco_bridge["test"],
    "nq": nq_bridge["test"],
    "lifestyle": lifestyle_bridge["test"],
    "recreation": recreation_bridge["test"],
    "science": science_bridge["test"],
    "technology": technology_bridge["test"],
    "writing": writing_bridge["test"]
}


for name, data in datasets_to_save.items():
    save_path = os.path.join(SAVE_ROOT, f"{name}.jsonl")
    data.to_json(save_path, force_ascii=False)
    print(f"✅ Saved {name} to {save_path}")