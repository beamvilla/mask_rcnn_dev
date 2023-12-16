import os
import sys
sys.path.append("./")
sys.path.append("./yolov5")

from src.yolov5.detect import YOLODetection


IMAGE_DIR = "./yolo_dataset_augment/images"
SUBSET = "test"

test_image_dir = os.path.join(IMAGE_DIR, SUBSET)

yolo_detection = YOLODetection(weight_path="./trained_models/yolo/best.pt")

for image_filename in os.listdir(test_image_dir):
    image_path = os.path.join(test_image_dir, image_filename)
    bbox, conf_scores, classes = yolo_detection.detect(
                                    image_path=image_path,
                                    image_size=[640, 640],
                                    conf_thres=0.5,
                                    iou_thres=0.5,
                                    max_det=10
                                )
    break