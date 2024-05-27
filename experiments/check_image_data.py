import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import List, Dict
from pathlib import Path
import sys
sys.path.append("./")

from src.utils.file_manager import load_json_file


def check_duplicate_train_image(
    main_images_dir: str, 
    another_images_dir: str 
) -> List[str]:
    duplicate_images_path = []

    for main_image_filename in tqdm(os.listdir(main_images_dir)):
        main_image_path = os.path.join(main_images_dir, main_image_filename)
        main_image_array = np.array(Image.open(main_image_path))

        for another_image_filename in os.listdir(another_images_dir):
            another_image_path = os.path.join(another_images_dir, another_image_filename)
            if main_image_path == another_image_path:
                continue

            another_image_array = np.array(Image.open(another_image_path))

            if np.array_equal(main_image_array, another_image_array):
                print("\nFound: " + another_image_path + "\n Dupplicated with: " + main_image_path)
                duplicate_images_path.append(another_image_path)
    
    return duplicate_images_path


def check_no_skin_object(labels_dir: str):
    for subset in SUBSETS:
        annotations_path = os.path.join(labels_dir, subset, f"{subset}.json")
        annotations = load_json_file(annotations_path)

        images_metadata = annotations["_via_img_metadata"]

        for _, metadata in images_metadata.items():
            if len(metadata["regions"]) > 0:
                found_skin = False
                n_skin = 0
                for region in metadata["regions"]:
                    if list(region["region_attributes"].values())[0] == "skin":
                        found_skin = True
                        n_skin += 1

                if not found_skin:
                    print(subset, " , ", metadata["filename"], " not found skin.")

                if n_skin > 1:
                    print(subset, " , ", metadata["filename"], "found multiply skins.")

def get_n_images(labels_dir: str):
    for subset in SUBSETS:
        annotations_path = os.path.join(labels_dir, subset, f"{subset}.json")
        annotations = load_json_file(annotations_path)

        images_metadata = annotations["_via_img_metadata"]
        n_image = 0
        for _, metadata in images_metadata.items():
            if len(metadata["regions"]) > 0:
                n_image += 1

        print(subset, " contains ", n_image, " images.")

def image_insight(dataset_dir: str):
    classes_map_path = os.path.join(dataset_dir, "classes_map.json")
    classes_map = load_json_file(classes_map_path)
    classes_obj_max = {}

    for classname in classes_map.keys():
        classes_obj_max[classname] = 0

    for folder in os.listdir(dataset_dir):
        folder_path = os.path.join(dataset_dir, folder)
        _folder_path = Path(folder_path)

        if not _folder_path.is_dir():
            continue

        for subset in SUBSETS:
            annotations_path = os.path.join(folder_path, "labels", subset, f"{subset}.json")
            annotations = load_json_file(annotations_path)

            images_metadata = annotations["_via_img_metadata"]
            for _, metadata in images_metadata.items():
                if len(metadata["regions"]) == 0:
                    continue
                
                classes_obj_count = {}
                for classname in classes_map.keys():
                    classes_obj_count[classname] = 0

                for region in metadata["regions"]:
                    class_obj_name = list(region["region_attributes"].values())[0]
                    classes_obj_count[class_obj_name] += 1
                
                if classes_obj_count[class_obj_name] > classes_obj_max[class_obj_name]:
                    classes_obj_max[class_obj_name] = classes_obj_count[class_obj_name]
    print(classes_obj_max)

def check_img_size(dataset_dir: str):
    width = []
    height = []
    for subset in SUBSETS:
        for folder in os.listdir(dataset_dir):
            folder_path = os.path.join(dataset_dir, folder)
            if not Path(folder_path).is_dir():
                continue

            images_dir = os.path.join(folder_path, "images", subset)

            for image_filename in tqdm(os.listdir(images_dir)):
                image_path = os.path.join(images_dir, image_filename)
                image = Image.open(image_path)
                w, h= image.size
                width.append(w)
                height.append(h)
    
    print(sorted(width))
    print(f"Max width : {max(width)}")
    print(f"Min width : {min(width)}")
    print(f"Mean width : {sum(width) / len(width)}\n")

    print(sorted(height))
    print(f"Max height : {max(height)}")
    print(f"Min height : {min(height)}")
    print(f"Mean height : {sum(height) / len(height)}\n")
                

SUBSETS = ["train", "test", "val"]
labels_dir = "./dataset/white_bg/labels"
#check_no_skin_object(labels_dir)
#get_n_images(labels_dir)

"""
check_duplicate_train_image(
    main_images_dir="./dataset/white_bg/images/test",
    another_images_dir="./dataset/white_bg/images/test"
)
"""

#image_insight("./dataset/23122023")
check_img_size("./only_defect")
