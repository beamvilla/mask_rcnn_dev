import os
import json
import tensorflow as tf
from datetime import datetime, timedelta

from .mrcnn import model as modellib
from .model_trainer.prepare_dataset import CustomDataset
from .custom_config import CustomConfig, TrainingConfig, dir_path


training_config = TrainingConfig()
mrcnn_config = CustomConfig()

def train():
    with open(training_config.CLASSES_MAP_PATH) as classesMap:
        classes_map = json.load(classesMap)

    weights_path = training_config.INITIALIZE_WEIGHT
    weight_type = "retrain"

    if training_config.INITIALIZE_WEIGHT in ["coco", "balloon", "mobile"]:
        weights_path = os.path.join(dir_path, f"models/mask_rcnn_{training_config.INITIALIZE_WEIGHT}.h5")
        weight_type = training_config.INITIALIZE_WEIGHT

    train_model_time = datetime.strftime(datetime.now() + timedelta(hours=7),"%Y%m%d_%H%M%S")
    save_trained_model_dir = os.path.join(dir_path, f"trained_models/{train_model_time}")
    checkpoint_name = f"{mrcnn_config.BACKBONE}_{weight_type}_{training_config.TRAINABLE_LAYERS}_{train_model_time}"
    save_history_path = f"{save_trained_model_dir}/{checkpoint_name}.csv"
    checkpoint_path = f"{save_trained_model_dir}/{checkpoint_name}"

    if not os.path.exists(save_trained_model_dir):
        os.makedirs(save_trained_model_dir)

    ## ======================= PREPARE DATASET =======================
    # Training dataset.
    dataset_train = CustomDataset()
    dataset_train.load_custom(training_config.DATASET_DIR, "train", classes_map)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = CustomDataset()
    dataset_val.load_custom(training_config.DATASET_DIR, "val", classes_map)
    dataset_val.prepare()

    ## ======================= CALLBACKS =======================
    callbacks_list = [
                            tf.keras.callbacks.CSVLogger(
                                save_history_path, 
                                separator=",", 
                                append=True
                            ),
                            tf.keras.callbacks.EarlyStopping(
                                monitor=training_config.EARLY_STOPPING_MONITOR,
                                patience=training_config.EARLY_STOPPING_PATIENCE,
                                mode=training_config.EARLY_STOPPING_MODE
                            )
                    ]

    ## ======================= LOAD MODEL =======================
    model = modellib.MaskRCNN(mode="training", 
                            config=mrcnn_config,
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

    ## ======================= TRAIN =======================
    model.train(
        dataset_train, 
        dataset_val,
        learning_rate=mrcnn_config.LEARNING_RATE,
        epochs=training_config.EPOCHS,
        layers=training_config.TRAINABLE_LAYERS,
        custom_callbacks=callbacks_list,
        checkpoint_name=checkpoint_path,
        checkpoint_monitor=training_config.MODEL_CHECKPOINT_MONITOR,
        checkpoint_mode=training_config.MODEL_CHECKPOINT_MODE
    )

    return save_trained_model_dir, checkpoint_name