import os
import cv2
import numpy as np
from numpy import random
from typing import List
from pathlib import Path
from tqdm import tqdm
import sys
sys.path.append("./")

from src.utils.file_manager import load_json_file, save_json_output

def convert_polygon_to_binary_mask(
    all_points_x: List[int], 
    all_points_y: List[int],
    height: int,
    width: int
) -> np.array:
    area = []
    for i in range(len(all_points_x)):
        area.append([all_points_x[i], all_points_y[i]])
    area = np.array(area)
    mask = np.zeros((height, width))
    cv2.fillPoly(mask, [area], 1)
    mask = mask.astype(bool)
    return mask


BG_DIR = "./bg"
DATASET_DIR = "./dataset/resized"
output_dir = os.path.join(DATASET_DIR, "bg_augment")

bg_image_paths = []

for bg_image_filename in os.listdir(BG_DIR):
    bg_image_paths.append(os.path.join(BG_DIR, bg_image_filename))

image_id_list = []
for folder in os.listdir(DATASET_DIR):
    folder_path = os.path.join(DATASET_DIR, folder)
    if not Path(folder_path).is_dir():
        continue

    images_dir = os.path.join(folder_path, "images", "train")
    labels_dir = os.path.join(folder_path, "labels", "train")

    annotations = load_json_file(os.path.join(labels_dir, "train.json"))
    images_metadata = annotations["_via_img_metadata"]

    try:
        bg_augment_annotations["_via_img_metadata"]
    except NameError:
        bg_augment_annotations = {}

        for via_keys, via_attr in annotations.items():
            if via_keys not in ["_via_img_metadata", "_via_image_id_list"]:
                bg_augment_annotations[via_keys] = via_attr
            else:
                bg_augment_annotations[via_keys] = {}

    for _, metadata in tqdm(images_metadata.items()):
        if len(metadata["regions"]) == 0:
            continue

        image_filename = metadata["filename"]

        if image_filename.startswith("augment"):
            continue

        image_path = os.path.join(images_dir, image_filename)
        image = cv2.imread(image_path)
        h, w, _ = image.shape
        
        for bg_id, bg_image_path in enumerate(bg_image_paths):
            bg_image = cv2.imread(bg_image_path)
            bg_image = cv2.resize(bg_image, (w, h))

            for region in metadata["regions"]:
                if list(region["region_attributes"].values())[0] == "skin":
                    all_points_x = region["shape_attributes"]["all_points_x"]
                    all_points_y = region["shape_attributes"]["all_points_y"]
                    break
            
            mask = convert_polygon_to_binary_mask(all_points_x, all_points_y, h, w)
            bg_image[mask] = image[mask]

            output_image_dir = os.path.join(output_dir, "images", "train")

            if not os.path.exists(output_image_dir):
                os.makedirs(output_image_dir)

            image_name, image_endswith = image_filename.split(".")
            augemnt_image_filename = "augment_bg_" + image_name + "_" + str(bg_id) +  "." + image_endswith
            output_image_path = os.path.join(output_image_dir, augemnt_image_filename)
            cv2.imwrite(output_image_path, bg_image)

            image_size = os.path.getsize(output_image_path)
            image_id = augemnt_image_filename + str(image_size)

            bg_augment_annotations["_via_img_metadata"][image_id] = {}
            bg_augment_annotations["_via_img_metadata"][image_id]["filename"] = augemnt_image_filename
            bg_augment_annotations["_via_img_metadata"][image_id]["size"] = image_size
            bg_augment_annotations["_via_img_metadata"][image_id]["regions"] = metadata["regions"]
            bg_augment_annotations["_via_img_metadata"][image_id]["file_attributes"] = metadata["file_attributes"]
            image_id_list.append(image_id)

bg_augment_annotations["_via_image_id_list"] = image_id_list
save_augmented_labels_dir = os.path.join(output_dir, "labels", "train")
if not os.path.exists(save_augmented_labels_dir):
    os.makedirs(save_augmented_labels_dir)

save_json_output(
    data=bg_augment_annotations,
    output_path=os.path.join(save_augmented_labels_dir, f"train.json")
)



