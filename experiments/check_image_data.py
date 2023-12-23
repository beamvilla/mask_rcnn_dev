import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import List, Dict
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

SUBSETS = ["test"]
labels_dir = "./dataset/white_bg/labels"
#check_no_skin_object(labels_dir)
#get_n_images(labels_dir)


check_duplicate_train_image(
    main_images_dir="./dataset/white_bg/images/test",
    another_images_dir="./dataset/white_bg/images/test"
)