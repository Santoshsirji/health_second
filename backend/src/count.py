from pathlib import Path
from collections import Counter

SCRIPT_DIR = Path(__file__).resolve().parent
train_dir = SCRIPT_DIR.parent / "data" / "chest_xray" / "train"

counts = {}
for class_folder in train_dir.iterdir():
    if class_folder.is_dir():
        counts[class_folder.name] = len(list(class_folder.glob("*.*")))  # counts all files

print(counts)
