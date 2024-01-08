import os
from pathlib import Path
from tqdm import tqdm
import numpy as np
import cv2
import sys
sys.path.append("./")

from src.utils.file_manager import load_json_file, save_json_output
from src.utils.bbox import polygon_to_rect


# Crop skin
# Remove skin annotations

SUBSETS = ["train", "test", "val"]
DATASET_DIR = "./dataset/resized"
OUTPUT_DIR = "./only_defect"
new_size = 480

for folder in os.listdir(DATASET_DIR):
    folder_path = os.path.join(DATASET_DIR, folder)
    
    if not Path(folder_path).is_dir():
        continue
    
    for subset in SUBSETS:
        images_dir = os.path.join(folder_path, "images", subset)
        labels_dir = os.path.join(folder_path, "labels", subset)
        labels_path = os.path.join(labels_dir, f"{subset}.json")

        annotations = load_json_file(labels_path)
        defect_annotations = {}

        for via_keys, via_attr in annotations.items():
            if via_keys not in ["_via_img_metadata", "_via_image_id_list"]:
                defect_annotations[via_keys] = via_attr
            else:
                defect_annotations[via_keys] = {}
        
        image_id_list = []
        for _, metadata in tqdm(annotations["_via_img_metadata"].items()):
            
            if len(metadata["regions"]) == 0:
                continue
            
            image_filename = metadata["filename"]
            image_path = os.path.join(images_dir, image_filename)
            image = cv2.imread(image_path)
            
            for region in metadata["regions"]:
                if list(region["region_attributes"].values())[0] == "skin":
                    all_x_skin = region["shape_attributes"]["all_points_x"]
                    all_y_skin = region["shape_attributes"]["all_points_y"]
                    break
            
            x_skin, y_skin, w_skin, h_skin = polygon_to_rect(all_x_skin, all_y_skin)
            image = image[y_skin: y_skin + h_skin, x_skin: x_skin + w_skin, :]
            h, w, _ = image.shape
            image = cv2.resize(image, (new_size, new_size))
            
            regions = []
            for region in metadata["regions"]:
                if list(region["region_attributes"].values())[0] == "skin":
                    continue

                all_point_x = (np.array(region["shape_attributes"]["all_points_x"]) - x_skin) * (new_size / w)
                all_point_x = all_point_x.astype(int)
                all_point_x = all_point_x.tolist()

                all_point_y = (np.array(region["shape_attributes"]["all_points_y"]) - y_skin) * (new_size / h)
                all_point_y = all_point_y.astype(int)
                all_point_y = all_point_y.tolist()

                regions.append(
                    {
                        "shape_attributes": {
                            "name": region["shape_attributes"]["name"],
                            "all_points_x": all_point_x,
                            "all_points_y": all_point_y
                        },
                        "region_attributes": region["region_attributes"]
                    }
                )
            
            output_image_dir = os.path.join(OUTPUT_DIR, folder, "images", subset)
            output_image_path = os.path.join(output_image_dir, image_filename)

            if not os.path.exists(output_image_dir):
                os.makedirs(output_image_dir)

            cv2.imwrite(output_image_path, image)
            image_size = os.path.getsize(output_image_path)
            image_id = image_filename + str(image_size)
        
            defect_annotations["_via_img_metadata"][image_id] = {}
            defect_annotations["_via_img_metadata"][image_id]["filename"] = image_filename
            defect_annotations["_via_img_metadata"][image_id]["size"] = image_size
            defect_annotations["_via_img_metadata"][image_id]["regions"] = regions
            defect_annotations["_via_img_metadata"][image_id]["file_attributes"] = metadata["file_attributes"]
            image_id_list.append(image_id)
        
        defect_annotations["_via_image_id_list"] = image_id_list
        save_labels_dir = os.path.join(OUTPUT_DIR, folder, "labels", subset)
        if not os.path.exists(save_labels_dir):
            os.makedirs(save_labels_dir)

        save_json_output(
            data=defect_annotations,
            output_path=os.path.join(save_labels_dir, f"{subset}.json")
        )
