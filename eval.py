import os
import json
import tensorflow as tf
import pandas as pd

from mrcnn import model as modellib
from evaluate.evaluate_model import evaluate
from evaluate.metrics import export_metric_result
from custom_config import EvalCoonfig, CustomConfig, dir_path


def eval(
    weight_path: str, 
    checkpoint_name: str, 
    eval_config_path: str,
    mrcnn_config_path: str
  ) -> None:
    
    eval_config = EvalCoonfig(eval_config_path)
    mrcnn_config = CustomConfig(mrcnn_config_path)
    
    CONFUSION_IDX_LIST = [
                            (0, 0), (0, 1), (0, 2), (0, 4),
                            (1, 0), (1, 1), (1, 2), (1, 3),
                            (2, 0), (2, 1), (2, 2), (2, 3),
                            (3, 1), (3, 2),
                            (4, 0)
                        ]

    COLORS = eval_config.COLORS
    DEVICE = eval_config.DEVICE
    IMAGE_DIR = eval_config.IMAGE_DIR
    LABEL_PATH = eval_config.LABEL_PATH
    CLASSES_MAP_PATH = eval_config.CLASSES_MAP_PATH
    IOU_THRESHOLD = eval_config.IOU_THRESHOLD
    MAX_BBOX_OVERLAP = eval_config.MAX_BBOX_OVERLAP

    ANNOTATIONS = json.load(open(LABEL_PATH))
    CLASSES_MAP = json.load(open(CLASSES_MAP_PATH))

    MODEL_DIR = ("/").join(weight_path.split("/")[:-1])
    SAVE_PRED_DIR = os.path.join(dir_path, f"{MODEL_DIR}/predictions")

    if not os.path.exists(SAVE_PRED_DIR):
       os.makedirs(SAVE_PRED_DIR)

    mrcnn_config.display()

    # Create model in inference mode
    with tf.device(DEVICE):
      model = modellib.MaskRCNN(
                                    mode="inference", 
                                    model_dir="./", 
                                    config=mrcnn_config
                                )

    print("\n\nLoading weights ... ")
    model.load_weights(weight_path, by_name=True)
    print("Weight loaded. \n\n")

    confusion_scores_dict = {idx: 0 for idx in CONFUSION_IDX_LIST}

    confusion_scores_dict = evaluate(
                              image_dir=IMAGE_DIR,
                              annotations=ANNOTATIONS,
                              confusion_scores_dict=confusion_scores_dict,
                              model=model,
                              classes_map=CLASSES_MAP,
                              colors=COLORS,
                              save_pred_dir=SAVE_PRED_DIR,
                              iou_threshold=IOU_THRESHOLD,
                              max_bbox_overlap=MAX_BBOX_OVERLAP
                            )
    confusion_df, recall_and_precision_metric = export_metric_result(
                                                      confusion_scores_dict=confusion_scores_dict,
                                                      metrics_list=["skin", "minor", "critical", "non_defect", "non_skin"],
                                                      classes_name=list(CLASSES_MAP.keys())
                                                    )

    confusion_df.to_csv(os.path.join(SAVE_PRED_DIR, f"{checkpoint_name}_confusion_matrix.csv"))
    recall_and_precision_metric.to_csv(os.path.join(SAVE_PRED_DIR, f"{checkpoint_name}_pr_metrics.csv"))

    print("\n\n========= PARAMS ===============")
    print(f"conf : {mrcnn_config.DETECTION_MIN_CONFIDENCE}")
    print(f"max bbox ooverlap : {MAX_BBOX_OVERLAP}")
    print(f"iou : {IOU_THRESHOLD}")

    print("\n\n========= CONFUSION MATRIX ===============")
    print(confusion_df)

    print("\n\n========= SCORES ===============")
    print(recall_and_precision_metric)
