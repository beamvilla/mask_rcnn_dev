import PIL
from PIL import Image
import os
import sys
from tqdm import tqdm
from typing import Dict, Optional, List

sys.path.append("./")

from src.utils.file_manager import load_json_file, save_json_output
from src.utils.augment import *


SOURCE_DIR = "./dataset/white_bg/"

images_dir = os.path.join(SOURCE_DIR, "images")
labels_dir = os.path.join(SOURCE_DIR, "labels")

subsets = ["train"]

BRIGHTNESS_THRESHOLDS = [0.8, 1.2]

AUGMENT_FUNCS = {
    "flip_hor": augment_flip_image_horizontal,
    "flip_vert": augment_flip_image_vertical,
    "brightness": augment_brightness
}

def process_augment(
    image: Image.Image, 
    image_filename: str,
    regions: List[Dict[object, object]],
    save_dir: str,
    augment_name: str,
    brightness_val: Optional[float] = None
) -> Tuple[str, List[Dict[str, Dict[str, object]]]]:
    
    augment_regions = []

    image_width, image_height = image.size
    image_size_maps = {
        "flip_hor": image_height,
        "flip_vert": image_width
    }

    _augment_func = AUGMENT_FUNCS[augment_name]


    if augment_name == "brightness":
        brightness_val_name = str(brightness_val).replace(".", "_")
        augmented_image_filename = f"augment_{augment_name}_{brightness_val_name}_" + image_filename
        augmented_image = _augment_func(image, brightness_val)
        augment_regions = regions
    else:
        for region in regions:
            shape_attr = region["shape_attributes"]
            region_attr = region["region_attributes"]

            polygon_maps = {
                    "flip_hor": "all_points_y",
                    "flip_vert": "all_points_x"
                }

            remain_polygons_key = None

            for flip, points in polygon_maps.items():
                if augment_name != flip:
                    remain_polygons_key = points

            augmented_image_filename = f"augment_{augment_name}_" + image_filename
            augmented_image, polygons = _augment_func(
                                                image=image,
                                                polygons=shape_attr[polygon_maps[augment_name]],
                                                image_size=image_size_maps[augment_name]
                                            )
            augmented_shape_attr = {
                "name": shape_attr["name"],
                polygon_maps[augment_name]: polygons,
                remain_polygons_key: shape_attr[remain_polygons_key]
            }
            augment_regions.append(
                {
                    "shape_attributes": augmented_shape_attr,
                    "region_attributes": region_attr
                }
            )
    augmented_image.save(os.path.join(save_dir, augmented_image_filename))

    return augmented_image_filename, augment_regions


for subset in subsets:
    save_augmented_image_dir = os.path.join(images_dir, "train_augment")

    if not os.path.exists(save_augmented_image_dir):
        os.makedirs(save_augmented_image_dir)

    subset_image_dir = os.path.join(images_dir, subset)
    annotations_path = os.path.join(labels_dir, subset, f"{subset}.json")
    annotations = load_json_file(annotations_path)

    images_metadata = annotations["_via_img_metadata"]

    add_augment_via = {}

    for via_keys, via_attr in annotations.items():
        if via_keys not in ["_via_img_metadata", "_via_image_id_list"]:
            add_augment_via[via_keys] = via_attr
        else:
            add_augment_via[via_keys] = {}
    
    all_image_ids = []
    for image_meta, metadata in tqdm(images_metadata.items()):
        if len(metadata["regions"]) > 0:
            image_filename = metadata["filename"]
            via_image_size = metadata["size"]
            regions = metadata["regions"]
            file_attributes = metadata["file_attributes"]

            # add original image
            image = Image.open(os.path.join(subset_image_dir, image_filename))

            all_image_ids.append(image_meta)
            add_augment_via["_via_img_metadata"][image_meta] = {}
            add_augment_via["_via_img_metadata"][image_meta]["filename"] = image_filename
            add_augment_via["_via_img_metadata"][image_meta]["size"] = via_image_size
            add_augment_via["_via_img_metadata"][image_meta]["regions"] = regions
            add_augment_via["_via_img_metadata"][image_meta]["file_attributes"] = file_attributes
            image.save(os.path.join(save_augmented_image_dir, image_filename))

            augment_iters = [1]
            for augment_name in ["brightness", "flip_hor", "flip_vert"]:
                if augment_name == "brightness":
                    augment_iters = BRIGHTNESS_THRESHOLDS
                
                for val in augment_iters:
                    brightness_val = None
                    if augment_name == "brightness":
                        brightness_val = val
                        
                    augment_image_filename, augment_regions = process_augment(
                                                                    image=image, 
                                                                    image_filename=image_filename,
                                                                    regions=regions,
                                                                    save_dir=save_augmented_image_dir,
                                                                    augment_name=augment_name,
                                                                    brightness_val=brightness_val
                                                                )
                    augment_image_size = os.path.getsize(os.path.join(save_augmented_image_dir, augment_image_filename))
                    augment_image_meta = augment_image_filename + str(augment_image_size)

                    all_image_ids.append(augment_image_meta)
                    add_augment_via["_via_img_metadata"][augment_image_meta] = {}
                    add_augment_via["_via_img_metadata"][augment_image_meta]["filename"] = augment_image_filename
                    add_augment_via["_via_img_metadata"][augment_image_meta]["size"] = augment_image_size
                    add_augment_via["_via_img_metadata"][augment_image_meta]["regions"] = augment_regions
                    add_augment_via["_via_img_metadata"][augment_image_meta]["file_attributes"] = file_attributes

    add_augment_via["_via_image_id_list"] = all_image_ids

    save_augmented_labels_dir = os.path.join(labels_dir, "train_augment")
    if not os.path.exists(save_augmented_labels_dir):
        os.makedirs(save_augmented_labels_dir)

    save_json_output(
        data=add_augment_via,
        output_path=os.path.join(save_augmented_labels_dir, f"{subset}.json")
    )


    
    

