from pathlib import Path
import os
import sys
import numpy as np
from tqdm import tqdm
from PIL import Image

sys.path.append("./")

from src.utils.file_manager import load_json_file, save_json_output


dataset_dir = "./dataset/23122023"
output_dir = "./dataset/resized"

SUBSETS = ["test", "train", "val"]

for folder in os.listdir(dataset_dir):
    folder_path = os.path.join(dataset_dir, folder)

    if not Path(folder_path).is_dir():
        continue
    
    for subset in SUBSETS:
        images_dir = os.path.join(folder_path, "images", subset)
        labels_dir = os.path.join(folder_path, "labels", subset)

        output_images_dir = os.path.join(output_dir, folder, "images", subset)
        output_labels_dir = os.path.join(output_dir, folder, "labels", subset)

        for o in [output_images_dir, output_labels_dir]:
            if not os.path.exists(o):
                os.makedirs(o)

        annotations = load_json_file(os.path.join(labels_dir, f"{subset}.json"))
        new_image_metadata = {}
        image_id_list = []

        for image_id, metadata in tqdm(annotations["_via_img_metadata"].items()):
            resized = False

            if len(metadata["regions"]) == 0:
                continue
            
            image_filename = metadata["filename"]
            image_size = metadata["size"]

            image_path = os.path.join(images_dir, image_filename)
            image = Image.open(image_path)
            w, h = image.size

            if w > 640:
                image = image.resize((640, 640))
                resized = True

            output_image_path = os.path.join(output_images_dir, image_filename)
            image.save(os.path.join(output_images_dir, image_filename))

            if resized:
                image_size = os.path.getsize(output_image_path)
                image_id = image_filename + str(image_size)

            image_id_list.append(image_id)
            new_image_metadata[image_id] = {}
            new_image_metadata[image_id]["filename"] = image_filename
            new_image_metadata[image_id]["size"] = image_size
            new_image_metadata[image_id]["file_attributes"] = metadata["file_attributes"]

            regions = metadata["regions"]
            

            if resized:
                regions = []
                for region in metadata["regions"]:
                    _region = {}
                    _region["shape_attributes"] =  {}
                    _region["shape_attributes"]["name"] = region["shape_attributes"]["name"]
                    _region["region_attributes"] = region["region_attributes"]

                    x = region["shape_attributes"]["all_points_x"]
                    y = region["shape_attributes"]["all_points_y"]

                    resized_x = np.array(x) * (640 / w)
                    resized_x= resized_x.astype(int)
                    resized_x = resized_x.tolist()

                    resized_y = np.array(y) * (640 / h)
                    resized_y= resized_y.astype(int)
                    resized_y = resized_y.tolist()

                    _region["shape_attributes"]["all_points_x"] = resized_x
                    _region["shape_attributes"]["all_points_y"] = resized_y
                    regions.append(_region)

            new_image_metadata[image_id]["regions"] = regions
        
        annotations["_via_img_metadata"] = new_image_metadata
        annotations["_via_image_id_list"] = image_id_list
        save_json_output(data=annotations, output_path=os.path.join(output_labels_dir, f"{subset}.json"))

