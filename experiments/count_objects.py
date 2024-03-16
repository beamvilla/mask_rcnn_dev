import os
import sys
sys.path.append("./")

from src.utils.file_manager import load_json_file, print_prettier_json


MAIN_DATASET_DIR = "./dataset/resized"
SUBSETS = ["train", "val", "test"]

objects_count = {}

for subset in SUBSETS:
    objects_count[subset] = {
        "non_augment": {
            "skin": 0,
            "minor": 0,
            "critical": 0
        },
        "augment": {
            "skin": 0,
            "minor": 0,
            "critical": 0
        }
    }


for dataset in os.listdir(MAIN_DATASET_DIR):
    if not os.path.isdir(os.path.join(MAIN_DATASET_DIR, dataset)):
        continue

    labels_dir = os.path.join(MAIN_DATASET_DIR, dataset, "labels")
    for subset in SUBSETS:
        labels_path = os.path.join(labels_dir, subset, f"{subset}.json")
        labels_file = load_json_file(labels_path)
        labels_metadata = labels_file["_via_img_metadata"]
        for image_metadata in labels_metadata.values():
            if image_metadata["filename"].startswith("augment"):
                augment_type = "augment"
            else:
                augment_type = "non_augment"

            for region in image_metadata["regions"]:
                object_class_name = list(region["region_attributes"].values())[0]
                objects_count[subset][augment_type][object_class_name] += 1

print_prettier_json(objects_count)