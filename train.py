import tensorflow as tf

import os
import json
import skimage.draw
from datetime import datetime, timedelta
import pandas as pd

from mrcnn.config import Config
from mrcnn import model as modellib

from model_trainer.prepare_dataset import CustomDataset


FOLDER_DATASET_NAME = "roseapple_new_dataset_05122021"
LABEL_DIR = "roseapple_new_dataset_label_16032023"

EPOCHS = 100
trainable_layers = "heads"
save_weight_period = 1
initialize_weight = "mobile"

train_label_path = f"{LABEL_DIR}/train/train_label.json"

with open(train_label_path) as annoFile:
  train_label = json.load(annoFile)

train_set_num = len(train_label)
print(f"[TRAIN_SET_NUM] : {train_set_num}")

with open("./trained_models/classes_map.json") as classesMap:
    classes_map = json.load(classesMap)


# Training dataset.
dataset_train = CustomDataset()
train_anno_dir = f"{LABEL_DIR}/train/train_label.json"
dataset_train.load_custom(FOLDER_DATASET_NAME, train_anno_dir, classes_map)
dataset_train.prepare()

# Validation dataset
dataset_val = CustomDataset()
val_anno_dir = f"{LABEL_DIR}/val/val_label.json"
dataset_val.load_custom(FOLDER_DATASET_NAME, val_anno_dir, classes_map)
dataset_val.prepare()


class CustomConfig(Config):
      """
          Configuration for training on the dataset.
          Derives from the base Config class and overrides some values.
      """
      # Give the configuration a recognizable name
      NAME = "object"
      BACKBONE = "mobilenetv1"
      RPN_ANCHOR_SCALES = (8 , 16, 32, 64, 128)

      # We use a GPU with 12GB memory, which can fit two images.
      # Adjust down if you use a smaller GPU.
      GPU_COUNT = 1 
      IMAGES_PER_GPU = 1

      # Number of classes (including background)
      NUM_CLASSES = len(classes_map) + 1  # Background + (desired classes)

      # Number of training steps per epoch
      STEPS_PER_EPOCH = 100
      # Skip detections with < 90% confidence
      DETECTION_MIN_CONFIDENCE = 0.7

      IMAGE_MAX_DIM = 384
      IMAGE_MIN_DIM = 384



train_model_time = datetime.strftime(datetime.now() + timedelta(hours=7),'%Y%m%d_%H%M%S')
save_trained_model_dir = f"trained_model/{train_model_time}"
save_trained_model_path = f"{save_trained_model_dir}/{CustomConfig.BACKBONE}_{train_model_time}.h5"
save_history_path = f"{save_trained_model_dir}/{CustomConfig.BACKBONE}_{train_model_time}.csv"

if not os.path.exists(save_trained_model_dir):
    os.makedirs(save_trained_model_dir)

callbacks_list = []
callbacks_list.append(
                        tf.keras.callbacks.CSVLogger(
                            save_history_path, 
                            separator=",", 
                            append=True
                        )
                    )
callbacks_list.append(
                        tf.keras.callbacks.EarlyStopping(
                            monitor="val_loss",
                            mode="auto"
                    )
                )


weights_path = initialize_weight

# Path to trained weights file
if initialize_weight in ["coco", "balloon", "mobile"]:
    weights_path = f"models/mask_rcnn_{initialize_weight}.h5"

config = CustomConfig()
model = modellib.MaskRCNN(mode="training", 
                          config=config,
                          model_dir=save_trained_model_dir)

model.load_weights(weights_path, 
                   by_name=True, 
                   exclude=[
                            "mrcnn_class_logits", 
                            "mrcnn_bbox_fc",
                            "mrcnn_bbox", 
                            "mrcnn_mask"
                            ]
                   )

model.train(
    dataset_train, 
    dataset_val,
    learning_rate=config.LEARNING_RATE,
    epochs=EPOCHS,
    layers=trainable_layers,
    custom_callbacks=callbacks_list
)
