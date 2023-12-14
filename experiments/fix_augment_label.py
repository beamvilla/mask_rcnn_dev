import json
from typing import Dict


correct_label_path = "./dataset/label/train/train.json"
incorrect_label_path = "./dataset/label/train_augment/train.json"

def extract_augment_method(filename: str) -> Dict[str, object]:
    split_filename = filename.split("_")
    