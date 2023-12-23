import os
import shutil
import sys
from tqdm import tqdm

sys.path.append("./")

from src.utils.file_manager import load_json_file


dataset_dir = "./dataset/black_bg"
label_dir = os.path.join(dataset_dir, "labels")
image_dir = os.path.join(dataset_dir, "images")

SUBSETS = ["train", "val", "test"]

for subset in SUBSETS:
    label_path = os.path.join(label_dir, subset, f"{subset}.json")
    image_metadata = load_json_file(label_path)["_via_img_metadata"]
    new_image_dir = os.path.join(image_dir, subset)

    if not os.path.exists(new_image_dir):
        os.makedirs(new_image_dir)

    for _, metadata in tqdm(image_metadata.items()):
        image_filename = metadata["filename"]
        image_path = os.path.join(image_dir, image_filename)
        new_image_path = os.path.join(new_image_dir, image_filename)
        shutil.move(src=image_path, dst=new_image_path)