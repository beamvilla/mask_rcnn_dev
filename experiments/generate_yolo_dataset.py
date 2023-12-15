from PIL import Image
import sys
import os

sys.path.append("./")

from src.utils.bbox import polygon_to_rect
from src.utils.file_manager import load_json_file, write_text_lines_file


ANNOTATIONS_PATH = "./dataset/label/val/val.json"
IMAGE_DIR = "./dataset/images"
OUTPUT_DIR  = "./yolo_dataset_augment"
SUBSET = "val"

image_output_dir = os.path.join(OUTPUT_DIR, "images", SUBSET)
label_output_dir = os.path.join(OUTPUT_DIR, "labels", SUBSET)

if not os.path.exists(image_output_dir):
    os.makedirs(image_output_dir)

if not os.path.exists(label_output_dir):
    os.makedirs(label_output_dir) 

annotations = load_json_file(ANNOTATIONS_PATH)["_via_img_metadata"]

classes_maps = {
    "skin": 0,
    "minor": 1,
    "critical": 1
}

for image_meta, metadata in annotations.items():
    labels = []
    image_path = os.path.join(IMAGE_DIR, metadata["filename"])
    image_name = metadata["filename"].split(".")[0]
    image = Image.open(image_path)
    w_image, h_image = image.size

    for region in metadata["regions"]:
        all_points_x = region["shape_attributes"]["all_points_x"]
        all_points_y = region["shape_attributes"]["all_points_y"]

        x, y, w, h = polygon_to_rect(polygon_x=all_points_x, polygon_y=all_points_y)

        # Normalize to yolo format
        x = (x + w / 2) / w_image
        y = (y + h / 2) / h_image
        w = w / w_image
        h = h / h_image

        class_name = list(region["region_attributes"].values())[0]
        class_id = classes_maps[class_name]

        labels.append(f"{class_id} {x} {y} {w} {h}\n")
    
    image.save(os.path.join(image_output_dir, metadata["filename"]))
    write_text_lines_file(lines=labels, output_path=os.path.join(label_output_dir, f"{image_name}.txt"))

print("Done.")
