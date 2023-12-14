import sys
import re
import os
import numpy as np
from PIL import Image
from typing import List

sys.path.append("./")

from src.utils.file_manager import load_json_file, save_json_output


def flip_vertical(image_width: int, x_polygon: List[int]) -> List[int]:
    all_points_x = np.array(x_polygon)
    all_points_x_flipped = image_width - all_points_x
    return all_points_x_flipped.tolist()

def flip_horizontal(image_height: int, y_polygon: List[int]) -> List[int]:
    all_points_y = np.array(y_polygon)
    all_points_y_flipped = image_height - all_points_y
    return all_points_y_flipped.tolist()


correct_label_path = "./dataset/label/train/train.json"
incorrect_label_path = "./dataset/label/train_augment/train.json"
images_dir = "./dataset/images"
output_dir = "./dataset/label/fixed_train_augment/"

correct_annotations = load_json_file(correct_label_path)
incorrect_annotations = load_json_file(incorrect_label_path)

correct_region_attributes = {}
for image_meta, metadata in correct_annotations["_via_img_metadata"].items():
    correct_region_attributes[metadata["filename"]] = metadata["regions"]

for image_meta, metadata in incorrect_annotations["_via_img_metadata"].items():
    image = Image.open(os.path.join(images_dir, metadata["filename"]))
    image_width, image_height = image.size

    source_image_meta = re.search(r"id.*", metadata["filename"]).group()
    region_attributes = correct_region_attributes[source_image_meta]

    if image_meta.startswith("augment_flip"):
        filpped_method = image_meta.split("_")[2]
        new_region_attributes = []
        for region_attr in region_attributes:
            _region_attr = {}
            _region_attr["shape_attributes"] = {}
            _region_attr["region_attributes"] = region_attr["region_attributes"]
            _region_attr["shape_attributes"]["name"] = region_attr["shape_attributes"]["name"]

            x = region_attr["shape_attributes"]["all_points_x"]
            y = region_attr["shape_attributes"]["all_points_y"]

            if filpped_method == "hor":
                _region_attr["shape_attributes"]["all_points_y"] = flip_horizontal(
                                                                    image_height=image_height, 
                                                                    y_polygon=y
                                                                )
                _region_attr["shape_attributes"]["all_points_x"] = region_attr["shape_attributes"]["all_points_x"]
            else:
                _region_attr["shape_attributes"]["all_points_x"] = flip_vertical(
                                                                    image_width=image_width, 
                                                                    x_polygon=x
                                                                )
                _region_attr["shape_attributes"]["all_points_y"] = region_attr["shape_attributes"]["all_points_y"]
            
            new_region_attributes.append(_region_attr)
        incorrect_annotations["_via_img_metadata"][image_meta]["regions"] = new_region_attributes
        continue

    incorrect_annotations["_via_img_metadata"][image_meta]["regions"] = region_attributes


if not os.path.exists(output_dir):
    os.makedirs(output_dir)


save_json_output(
    data=incorrect_annotations,
    output_path=os.path.join(output_dir, "train.json")
)
