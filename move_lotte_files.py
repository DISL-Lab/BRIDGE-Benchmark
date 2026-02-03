import os
import shutil
from pathlib import Path

categories = ['lifestyle', 'recreation', 'technology', 'science', 'writing']

for category in categories:
    source_dir = Path(f'./data/lotte/{category}/test')
    target_dir = Path(f'./data/{category}/test')
    
    if not source_dir.exists():
        print(f"⚠️  Source not found: {source_dir}")
        continue
    
    target_dir.mkdir(parents=True, exist_ok=True)
    
    files = [f for f in source_dir.glob('*') if f.is_file()]
    
    if not files:
        print(f"📁 No files in {source_dir}")
        continue
    
    print(f"📦 {category}: moving {len(files)} files...")
    
    for file_path in files:
        target_path = target_dir / file_path.name
        
        if target_path.exists():
            print(f"  ⚠️  Skip (exists): {file_path.name}")
            continue
        
        shutil.move(str(file_path), str(target_path))
        print(f"  ✓ {file_path.name}")

print("\n✅ Done!")