import tensorflow as tf
import os
import json
import skimage.draw

from custom_config import EvalCoonfig, CustomConfig
from mrcnn import model as modellib


weight_path = "./trained_models/20230404_123755/mobilenetv1_mobile_heads_20230404_123755_0000.h5"
weight_dir = "/".join(weight_path.split("/")[:-1])
eval_config_path = os.path.join(weight_dir, "config", "eval.json")
mrcnn_config_path = os.path.join(weight_dir, "config", "mrcnn_config.json")

eval_config = EvalCoonfig(eval_config_path)
mrcnn_config = CustomConfig(mrcnn_config_path)

COLORS = eval_config.COLORS
DEVICE = eval_config.DEVICE
IMAGE_DIR = eval_config.IMAGE_DIR
LABEL_PATH = eval_config.LABEL_PATH
CLASSES_MAP_PATH = eval_config.CLASSES_MAP_PATH
TEST_SET = eval_config.TEST_SET
IOU_THRESHOLD = eval_config.IOU_THRESHOLD

MODEL_DIR = ("/").join(weight_path.split("/")[:-1])
SAVE_PRED_DIR = f"{MODEL_DIR}/predictions"

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

image_path = "./roseapple_new_dataset_05122021/id_6_side_2.jpg"
gt_image = skimage.io.imread(image_path)
pred_results = model.detect([gt_image], verbose=1)
pred_results = pred_results[0]
print(pred_results)
