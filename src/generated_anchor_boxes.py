import os
import torch
from typing import List, Tuple, Dict, Optional

from utils.bbox import polygon_to_rect, xywh_to_x1y1x2y2
from utils.file_manager import load_json_file
from anchors_utils.boxes_utils import get_redefined_bbox
from anchors_utils.clustering_boxes import KMeans_clustering_anchor_boxes
from anchors_utils.distance_function import IoU


def generate_redefine_bbox(
    target_obj_number: Optional[int] = None
) -> List[Tuple[int, int, int, int]]:
    counter = {
        "skin": 0,
        "minor": 0,
        "critical": 0
    }
    DATASET_DIR = "./dataset/resized"
    sub_dirs = os.listdir(DATASET_DIR)
    subset = "train"
    redefined_bboxes = []
    for sub_dir in sub_dirs:
        sub_dir = os.path.join(DATASET_DIR, sub_dir)
        if not os.path.isdir(sub_dir):
            continue

        images_dir = os.path.join(sub_dir, "images", subset)
        labels_path = os.path.join(sub_dir, "labels", subset, f"{subset}.json")
        labels = load_json_file(labels_path)["_via_img_metadata"]

        for _, image_meta in labels.items():
            image_path = os.path.join(images_dir, image_meta["filename"])
            for region in image_meta["regions"]:
                x = region["shape_attributes"]["all_points_x"]
                y = region["shape_attributes"]["all_points_y"]
                class_name = list(region["region_attributes"].values())[0]

                bbox = polygon_to_rect(polygon_x=x, polygon_y=y)  # x, y, w, h
                bbox = xywh_to_x1y1x2y2(bbox)

                if target_obj_number is not None:
                    if counter[class_name] == target_obj_number:
                        continue
                    counter[class_name] += 1
                    redefined_bboxes.append(bbox)
                else:
                    redefined_bboxes.append(bbox)
    return redefined_bboxes


# Start clusering
redefined_bboxes = generate_redefine_bbox(target_obj_number=None)
redefined_bboxes = torch.tensor(redefined_bboxes, dtype=torch.float)
anchors, distances = KMeans_clustering_anchor_boxes(redefined_bboxes, k=15, stop_iter=5)
anchors_width_height_cluster = []

for anchor in anchors:
    x1, y1, x2, y2 = anchor.tolist()
    anchors_width_height_cluster.append([int(x2 - x1), int(y2 - y1)])

print(anchors_width_height_cluster)
