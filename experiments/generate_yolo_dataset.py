from PIL import Image
import sys
import os
from typing import Dict
from tqdm import tqdm

sys.path.append("./")

from src.utils.bbox import polygon_to_rect
from src.utils.file_manager import load_json_file, write_text_lines_file


def convert(annotations: Dict[object, object], mask=False) -> None:
    for image_meta, metadata in tqdm(annotations.items()):
        labels = []
        image_path = os.path.join(IMAGE_DIR, metadata["filename"])
        image_name = metadata["filename"].split(".")[0]
        image = Image.open(image_path)
        w_image, h_image = image.size

        if len(metadata["regions"]) == 0:
            continue

        for region in metadata["regions"]:
            all_points_x = region["shape_attributes"]["all_points_x"]
            all_points_y = region["shape_attributes"]["all_points_y"]

            if not mask:
                x, y, w, h = polygon_to_rect(polygon_x=all_points_x, polygon_y=all_points_y)

                # Normalize to yolo format
                x = (x + w / 2) / w_image
                y = (y + h / 2) / h_image
                w /= w_image
                h /= h_image
            else:
                x = [x / w_image for x in all_points_x]
                y = [y / h_image for y in all_points_y]

            class_name = list(region["region_attributes"].values())[0]
            class_id = CLASSES_MAP[class_name]

            if not mask:
                labels.append(f"{class_id} {x} {y} {w} {h}\n")
            else:
                _label = str(class_id)
                for i in range(len(x)):
                    point_x = x[i]
                    point_y = y[i]
                    _label += f" {point_x} {point_y}"
                _label += "\n"
                labels.append(_label)

        image.save(os.path.join(image_output_dir, metadata["filename"]))
        write_text_lines_file(lines=labels, output_path=os.path.join(label_output_dir, f"{image_name}.txt"))

    print("Done.")


SUBSET = "test"
ANNOTATIONS_PATH = f"./dataset/white_bg/labels/test/{SUBSET}.json"
IMAGE_DIR = "./dataset/white_bg/images/test"
OUTPUT_DIR  = "./yolo_dataset_augment_3_classes_mask"
CLASSES_MAP = {
    "skin"      : 0,
    "minor"     : 1,
    "critical"  : 2
}

image_output_dir = os.path.join(OUTPUT_DIR, "images", SUBSET)
label_output_dir = os.path.join(OUTPUT_DIR, "labels", SUBSET)

if not os.path.exists(image_output_dir):
    os.makedirs(image_output_dir)

if not os.path.exists(label_output_dir):
    os.makedirs(label_output_dir) 

annotations = load_json_file(ANNOTATIONS_PATH)["_via_img_metadata"]
convert(annotations=annotations, mask=True)
