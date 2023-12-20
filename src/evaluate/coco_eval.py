import os
from typing import Dict

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def eval(
    ground_truth_path: str, 
    detection_path: str,
    test_image_dir: str,
    classes_maps: Dict[int, str],
    detection_type: str = "bbox"
) -> Dict[str, Dict[str, float]]:
    
    if detection_type not in ["segm", "bbox"]:
        raise ValueError(f"Not supported {detection_type}")
    
    coco_ground_truth = COCO(ground_truth_path)
    coco_detection = COCO(detection_path)

    # running evaluation
    cocoEval = COCOeval(coco_ground_truth, coco_detection, "bbox")

    image_files = os.listdir(test_image_dir)
    n_image = len(image_files)

    AP_scores = {}
    class_ids = []
    for class_id, class_name in classes_maps.items():
        AP_scores[class_name] = {
            "iou_50_to_95": 0.0,
            "iou_50": 0.0,
            "iou_75": 0.0,
            "n_image": 0
        }
        class_ids.append(class_id + 1)

    for class_id in class_ids:
        image_ID = 1
        while image_ID < n_image + 1:
            print("\n\n", image_files[image_ID - 1], " id: ", image_ID)
            cocoEval.params.catIds = [class_id]
            cocoEval.params.imgIds  = image_ID
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()

            if cocoEval.stats[0] != -1 and cocoEval.stats[1] != -1 and cocoEval.stats[2] != -1:
                AP_scores[classes_maps[class_id - 1]]["iou_50_to_95"] += cocoEval.stats[0]
                AP_scores[classes_maps[class_id - 1]]["iou_50"] += cocoEval.stats[1]
                AP_scores[classes_maps[class_id - 1]]["iou_75"] += cocoEval.stats[2]
                AP_scores[classes_maps[class_id - 1]]["n_image"] += 1

            image_ID += 1
        for k in ["iou_50_to_95", "iou_50", "iou_75"]:
            AP_scores[classes_maps[class_id - 1]][k] /= AP_scores[classes_maps[class_id - 1]]["n_image"]
    return AP_scores