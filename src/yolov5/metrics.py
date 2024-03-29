import numpy as np
import pandas as pd
from typing import Dict, List, Union


def box_iou_calc(boxes1: np.array, boxes2: np.array) -> np.array:
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        boxes1 (Array[N, 4])
        boxes2 (Array[M, 4])
    Returns:
        iou (Array[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2

    This implementation is taken from the above link and changed so that it only uses numpy..
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(boxes1.T)
    area2 = box_area(boxes2.T)

    lt = np.maximum(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    inter = np.prod(np.clip(rb - lt, a_min=0, a_max=None), 2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


class ConfusionMatrix:
    def __init__(
        self, 
        num_classes: int, 
        conf_thres=0.3, 
        iou_thres=0.5
    ) -> None:
        self.matrix = np.zeros((num_classes + 1, num_classes + 1))
        self.num_classes = num_classes
        self.CONF_THRESHOLD = conf_thres
        self.IOU_THRESHOLD = iou_thres

    def process_batch(
        self, 
        detections: Dict[str, np.array], 
        labels: Dict[str, np.array]
    ) -> None:
        """
        Return intersection-over-union (Jaccard index) of boxes.
        Both sets of boxes are expected to be in (x1, y1, x2, y2) format. >> int
        Arguments:
            detections:
                {
                    "boxes": np.array([[x1, y1, x2, y2]]),
                    "classes": np.array([int]),
                    "masks": np.array([[binary masks]])
                }
            labels:
                {
                    "boxes": np.array([[x1, y1, x2, y2]]),
                    "classes": np.array([int]),
                    "masks": np.array([[binary masks]])
                }

        Returns:
            None, updates confusion matrix accordingly
        """
        gt_classes = labels["classes"].astype(np.int16)
        gt_boxes = labels["boxes"]
    
        detection_boxes = detections["boxes"]
        detection_classes = detections["classes"].astype(np.int16)

        if len(detection_boxes) == 0:
            # detections are empty, end of process
            for i in range(len(gt_boxes)):
                gt_class = gt_classes[i]
                self.matrix[self.num_classes, gt_class] += 1
            return

        all_ious = box_iou_calc(gt_boxes, detection_boxes)

        want_idx = np.where(all_ious > self.IOU_THRESHOLD)

        all_matches = [[want_idx[0][i], want_idx[1][i], all_ious[want_idx[0][i], want_idx[1][i]]]
                       for i in range(want_idx[0].shape[0])]

        all_matches = np.array(all_matches)
        if all_matches.shape[0] > 0:  # if there is match
            all_matches = all_matches[all_matches[:, 2].argsort()[::-1]]

            all_matches = all_matches[np.unique(all_matches[:, 1], return_index=True)[1]]

            all_matches = all_matches[all_matches[:, 2].argsort()[::-1]]

            all_matches = all_matches[np.unique(all_matches[:, 0], return_index=True)[1]]

        for i in range(len(gt_boxes)):
            gt_class = gt_classes[i]
            if all_matches.shape[0] > 0 and all_matches[all_matches[:, 0] == i].shape[0] == 1:
                detection_class = detection_classes[int(all_matches[all_matches[:, 0] == i, 1][0])]
                self.matrix[detection_class, gt_class] += 1
            else:
                self.matrix[self.num_classes, gt_class] += 1

        for i, _ in enumerate(detection_boxes):
            if not all_matches.shape[0] or (all_matches.shape[0] and all_matches[all_matches[:, 1] == i].shape[0] == 0):
                detection_class = detection_classes[i]
                self.matrix[detection_class, self.num_classes] += 1

    def return_matrix(self):
        return self.matrix

    def print_matrix(self):
        for i in range(self.num_classes + 1):
            print(' '.join(map(str, self.matrix[i])))

    def get_confusion_matrix_df(self, classes_map_ids: Dict[int, str]) -> pd.DataFrame:
        class_columns = []

        for i in range(len(classes_map_ids)):
            class_columns.append(classes_map_ids[i])

        class_columns.append("bg")
        confusion_matrix_df = pd.DataFrame(columns=class_columns, index=class_columns)

        for i in range(len(class_columns)):
            try:
                class_col = classes_map_ids[i]
            except KeyError:
                class_col = "bg"
            confusion_matrix_df.loc[:, class_col] = self.matrix[:, i].astype(int)
        return confusion_matrix_df
    
    def get_metric_scores(self, classes_map_ids: Dict[int, str]) -> Dict[str, Dict[str, float]]:
        metric_results = {}

        for _, class_name in classes_map_ids.items():
            metric_results[class_name] = {"precision": 0.0, "recall": 0.0}

        for i in range(len(self.matrix)):
            tp = self.matrix[i, i]

            precision = tp / sum(self.matrix[i, :])
            recall = tp / sum(self.matrix[:, i])

            try:
                metric_results[classes_map_ids[i]]["precision"] = precision
            except KeyError:
                continue
            
            metric_results[classes_map_ids[i]]["recall"] = recall
        return metric_results
