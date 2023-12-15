import os
from PIL import Image
import numpy as np
from tqdm import tqdm
import sys
sys.path.append("./")

from src.utils.bbox import polygon_to_rect
from src.utils.file_manager import load_json_file


OUTPUT_DIR = "./defect_type_dataset"
SUBSET = "test"

ANNOTATIONS_PATH = "./dataset/label/test/test.json"
IMAGE_DIR = IMAGE_DIR = "./dataset/images"
classes_name = ["minor", "critical"]

for c in classes_name:
    label_output_dir = os.path.join(OUTPUT_DIR, SUBSET, c)

    if not os.path.exists(label_output_dir):
        os.makedirs(label_output_dir)

annotations = load_json_file(ANNOTATIONS_PATH)["_via_img_metadata"]

for image_meta, metadata in tqdm(annotations.items()):
    labels = []
    image_path = os.path.join(IMAGE_DIR, metadata["filename"])
    image_name = metadata["filename"].split(".")[0]

    image = np.array(Image.open(image_path))
    image_cnt = 0

    for region in metadata["regions"]:
        class_name = list(region["region_attributes"].values())[0]
        class_dir = os.path.join(OUTPUT_DIR, SUBSET, class_name)

        if class_name == "skin":
            continue

        all_points_x = region["shape_attributes"]["all_points_x"]
        all_points_y = region["shape_attributes"]["all_points_y"]

        x, y, w, h = polygon_to_rect(polygon_x=all_points_x, polygon_y=all_points_y)

        class_image = Image.fromarray(image[y: y + h, x: x + w, :])
        class_image.save(os.path.join(class_dir, f"{image_name}_{image_cnt}.jpg"))
        image_cnt += 1



