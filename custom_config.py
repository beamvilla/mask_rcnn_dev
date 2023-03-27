import json

from mrcnn.config import Config


with open("./config/mrcnn_config.json") as configFile:
      MRCNN_CONFIG = json.load(configFile)

with open("./config/training.json") as configFile:
      TRAINING_CONFIG = json.load(configFile)

with open("./config/eval.json") as configFile:
      EVAL_CONFIG = json.load(configFile)

class CustomConfig(Config):
      """
          Configuration for training on the dataset.
          Derives from the base Config class and overrides some values.
      """
      # Give the configuration a recognizable name
      NAME = MRCNN_CONFIG["NAME"]
      BACKBONE = MRCNN_CONFIG["BACKBONE"]
      RPN_ANCHOR_SCALES = tuple(MRCNN_CONFIG["RPN_ANCHOR_SCALES"])

      MAX_GT_INSTANCES = MRCNN_CONFIG["MAX_GT_INSTANCES"]
      DETECTION_MAX_INSTANCES = MRCNN_CONFIG["DETECTION_MAX_INSTANCES"]

      POST_NMS_ROIS_TRAINING = MRCNN_CONFIG["POST_NMS_ROIS_TRAINING"]
      POST_NMS_ROIS_INFERENCE = MRCNN_CONFIG["POST_NMS_ROIS_INFERENCE"]
      
      RPN_TRAIN_ANCHORS_PER_IMAGE = MRCNN_CONFIG["RPN_TRAIN_ANCHORS_PER_IMAGE"]
      TRAIN_ROIS_PER_IMAGE = MRCNN_CONFIG["TRAIN_ROIS_PER_IMAGE"]

      USE_MULTIPROCESSING = MRCNN_CONFIG["USE_MULTIPROCESSING"]
      IMAGE_RESIZE_MODE = MRCNN_CONFIG["IMAGE_RESIZE_MODE"]

      # We use a GPU with 12GB memory, which can fit two images.
      # Adjust down if you use a smaller GPU.
      GPU_COUNT = MRCNN_CONFIG["GPU_COUNT"]
      IMAGES_PER_GPU = MRCNN_CONFIG["IMAGES_PER_GPU"]

      # Number of classes (including background)
      NUM_CLASSES = MRCNN_CONFIG["NUM_CLASSES"] + 1  # Background + (desired classes)

      # Number of training steps per epoch
      STEPS_PER_EPOCH = MRCNN_CONFIG["STEPS_PER_EPOCH"]

      # Skip detections with < xx% confidence
      DETECTION_MIN_CONFIDENCE = MRCNN_CONFIG["DETECTION_MIN_CONFIDENCE"]

      IMAGE_MAX_DIM = MRCNN_CONFIG["IMAGE_MAX_DIM"]
      IMAGE_MIN_DIM = MRCNN_CONFIG["IMAGE_MIN_DIM"]


class TrainingConfig:
      DATASET_DIR = TRAINING_CONFIG["dataset"]["dataset_dir"]
      LABEL_DIR = TRAINING_CONFIG["dataset"]["dataset_dir"]

      TRAINABLE_LAYERS = TRAINING_CONFIG["model"]["trainable_layers"]
      INITIALIZE_WEIGHT = TRAINING_CONFIG["model"]["initialize_weight"]

      EPOCHS = TRAINING_CONFIG["params"]["epochs"]

      EARLY_STOPPING_MONITOR = TRAINING_CONFIG["params"]["early_stopping_monitor"]
      EARLY_STOPPING_PATIENCE = TRAINING_CONFIG["params"]["early_stopping_patience"]
      EARLY_STOPPING_MODE = TRAINING_CONFIG["params"]["early_stopping_mode"]

      MODEL_CHECKPOINT_MONITOR = TRAINING_CONFIG["params"]["model_checkpoint_monitor"]
      MODEL_CHECKPOINT_MODE = TRAINING_CONFIG["params"]["model_checkpoint_mode"]

class EvalCoonfig:
      COLORS = EVAL_CONFIG["mask_color"]
      for obj, color in COLORS.items():
            COLORS[obj] = tuple(color)

      DEVICE = EVAL_CONFIG["device"]
      IMAGE_DIR = EVAL_CONFIG["image_dir"]
      LABEL_DIR = EVAL_CONFIG["label_dir"]
      CLASSES_MAP_PATH = EVAL_CONFIG["classes_map_path"]
      TEST_SET = EVAL_CONFIG["test_set"]
      IOU_THRESHOLD = EVAL_CONFIG["iou_threshold"]


mrcnn_config = CustomConfig()
training_config = TrainingConfig()
eval_config = EvalCoonfig()