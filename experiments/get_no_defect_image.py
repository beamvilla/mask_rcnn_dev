import os
import sys
from typing import Dict
from tqdm import tqdm
import shutil

sys.path.append("./")

from src.utils.file_manager import load_json_file


def check(annotations: Dict[object, object]) -> Dict[str, int]:
    filenames = {}
    for _, metadata in tqdm(annotations.items()):
        is_only_skin = True

        for region in metadata["regions"]:
            class_name = list(region["region_attributes"].values())[0]
            
            if class_name != "skin":
                is_only_skin = False
                break
        
        if is_only_skin:
            filenames[metadata["filename"]] = 1
    return filenames


SUBSET = "train"
ANNOTATIONS_PATH = f"./dataset/label/fixed_train_augment/{SUBSET}.json"
IMAGE_DIR = "./dataset/images"
OUTPUT_DIR  = "./yolo_dataset_augment_2_classes_mask"
CLASSES_MAP = {
    "skin"      : 0,
    "minor"     : 1,
    "critical"  : 1
}

image_output_dir = os.path.join(OUTPUT_DIR, "images", SUBSET)
label_output_dir = os.path.join(OUTPUT_DIR, "labels", SUBSET)

if not os.path.exists(image_output_dir):
    os.makedirs(image_output_dir)

if not os.path.exists(label_output_dir):
    os.makedirs(label_output_dir) 

annotations = load_json_file(ANNOTATIONS_PATH)["_via_img_metadata"]
skin_image_filenames = check(annotations=annotations)

images_dir = "./skin_detection/train"
abnormal_dir = os.path.join(images_dir, "abnormal")
normal_dir = os.path.join(images_dir, "normal")

for path in [abnormal_dir, normal_dir]:
    if not os.path.exists(path):
        os.makedirs(path)

for f in os.listdir(images_dir):
    if f.endswith(".jpg"):
        try:
            skin_image_filenames[f]
        except KeyError:
            shutil.move(os.path.join(images_dir, f), os.path.join(abnormal_dir, f))
            continue
        
        shutil.move(os.path.join(images_dir, f), os.path.join(normal_dir, f))

