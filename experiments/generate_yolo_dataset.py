from PIL import Image
import sys
import os
from pathlib import Path
from typing import Dict
from tqdm import tqdm

sys.path.append("./")

from src.utils.bbox import polygon_to_rect
from src.utils.file_manager import load_json_file, write_text_lines_file


def convert(
    annotations: Dict[object, object], 
    images_dir: str,
    image_output_dir: str,
    label_output_dir: str,
    mask=False
) -> None:
    for _, metadata in tqdm(annotations.items()):
        labels = []
        image_path = os.path.join(images_dir, metadata["filename"])
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


SUBSETS = ["train", "val", "test"]
DATASET_DIR = "./dataset/resized/"
OUTPUT_DIR  = "./yolo_dataset_augment_3_classes_mask"
CLASSES_MAP = {
    "skin"      : 0,
    "minor"     : 1,
    "critical"  : 2
}

for subset in SUBSETS:
    image_output_dir = os.path.join(OUTPUT_DIR, "images", subset)
    label_output_dir = os.path.join(OUTPUT_DIR, "labels", subset)

    for o in [image_output_dir, label_output_dir]:
        if not os.path.exists(o):
            os.makedirs(o)
    
    for folder in os.listdir(DATASET_DIR):
        folder_path = os.path.join(DATASET_DIR, folder)

        if not Path(folder_path).is_dir():
            continue
    
        images_dir = os.path.join(folder_path, "images", subset)
        labels_dir = os.path.join(folder_path, "labels", subset)

        annotations = load_json_file(os.path.join(labels_dir, f"{subset}.json"))["_via_img_metadata"]
        convert(
            annotations=annotations, 
            images_dir=images_dir,
            label_output_dir=label_output_dir,
            image_output_dir=image_output_dir,
            mask=True
        )
