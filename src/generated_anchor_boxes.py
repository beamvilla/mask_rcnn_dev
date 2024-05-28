import os
import cv2
import torch

from utils.bbox import polygon_to_rect, xywh_to_x1y1x2y2
from utils.file_manager import load_json_file
from anchors_utils.boxes_utils import get_redefined_bbox
from anchors_utils.clustering_boxes import KMeans
from anchors_utils.distance_function import IoU


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
        #image = cv2.imread(image_path)
        for region in image_meta["regions"]:
            x = region["shape_attributes"]["all_points_x"]
            y = region["shape_attributes"]["all_points_y"]
            class_name = list(region["region_attributes"].values())[0]

            if class_name == "critical":
                bbox = polygon_to_rect(polygon_x=x, polygon_y=y)  # x, y, w, h
                bbox = xywh_to_x1y1x2y2(bbox)
                redefined_bboxes.append(bbox)
            #redefined_bbox = get_redefined_bbox(image=image, bbox=bbox, base=640)

# Start clusering
redefined_bboxes = torch.tensor(redefined_bboxes, dtype=torch.float)
anchors, distances = KMeans(redefined_bboxes, k=5)
anchors_width_height_cluster = []

for anchor in anchors:
    x1, y1, x2, y2 = anchor.tolist()
    anchors_width_height_cluster.append([int(x2 - x1), int(y2 - y1)])
print(anchors_width_height_cluster)
            

