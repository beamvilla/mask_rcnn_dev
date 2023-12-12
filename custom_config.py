import json
import numpy as np

from mrcnn.config import Config


with open("./config/eval.json") as configFile:
      EVAL_CONFIG = json.load(configFile)

class CustomConfig(Config):
      def __init__(self, config_path="./config/mrcnn_config.json"):
            with open(config_path) as configFile:
                  MRCNN_CONFIG = json.load(configFile)
            
            """
            Configuration for training on the dataset.
            Derives from the base Config class and overrides some values.
            """
            # Give the configuration a recognizable name
            self.NAME = MRCNN_CONFIG["NAME"]
            self.BACKBONE = MRCNN_CONFIG["BACKBONE"]
            self.RPN_ANCHOR_SCALES = tuple(MRCNN_CONFIG["RPN_ANCHOR_SCALES"])

            self.MAX_GT_INSTANCES = MRCNN_CONFIG["MAX_GT_INSTANCES"]
            self.DETECTION_MAX_INSTANCES = MRCNN_CONFIG["DETECTION_MAX_INSTANCES"]

            self.POST_NMS_ROIS_TRAINING = MRCNN_CONFIG["POST_NMS_ROIS_TRAINING"]
            self.POST_NMS_ROIS_INFERENCE = MRCNN_CONFIG["POST_NMS_ROIS_INFERENCE"]
            
            self.RPN_TRAIN_ANCHORS_PER_IMAGE = MRCNN_CONFIG["RPN_TRAIN_ANCHORS_PER_IMAGE"]
            self.TRAIN_ROIS_PER_IMAGE = MRCNN_CONFIG["TRAIN_ROIS_PER_IMAGE"]

            self.USE_MULTIPROCESSING = MRCNN_CONFIG["USE_MULTIPROCESSING"]
            self.IMAGE_RESIZE_MODE = MRCNN_CONFIG["IMAGE_RESIZE_MODE"]

            # We use a GPU with 12GB memory, which can fit two images.
            # Adjust down if you use a smaller GPU.
            self.GPU_COUNT = MRCNN_CONFIG["GPU_COUNT"]
            self.IMAGES_PER_GPU = MRCNN_CONFIG["IMAGES_PER_GPU"]

            # Number of classes (including background)
            self.NUM_CLASSES = MRCNN_CONFIG["NUM_CLASSES"] + 1  # Background + (desired classes)

            # Number of training steps per epoch
            self.STEPS_PER_EPOCH = MRCNN_CONFIG["STEPS_PER_EPOCH"]

            # Skip detections with < xx% confidence
            self.DETECTION_MIN_CONFIDENCE = MRCNN_CONFIG["DETECTION_MIN_CONFIDENCE"]

            self.IMAGE_MAX_DIM = MRCNN_CONFIG["IMAGE_MAX_DIM"]
            self.IMAGE_MIN_DIM = MRCNN_CONFIG["IMAGE_MIN_DIM"]

            """Set values of computed attributes."""
            # Effective batch size
            self.BATCH_SIZE = self.IMAGES_PER_GPU * self.GPU_COUNT

            # Input image size
            if self.IMAGE_RESIZE_MODE == "crop":
                  self.IMAGE_SHAPE = np.array([self.IMAGE_MIN_DIM, self.IMAGE_MIN_DIM, 3])
            else:
                  self.IMAGE_SHAPE = np.array([self.IMAGE_MAX_DIM, self.IMAGE_MAX_DIM, 3])

            # Image meta data length
            # See compose_image_meta() for details
            self.IMAGE_META_SIZE = 1 + 3 + 3 + 4 + 1 + self.NUM_CLASSES

            # Add Resolution to config name
            self.NAME = "{}_".format(self.IMAGE_MAX_DIM)+self.NAME
      


class TrainingConfig:
      def __init__(self, config_path="./config/training.json"):
            with open(config_path) as configFile:
                  TRAINING_CONFIG = json.load(configFile)

            self.DATASET_DIR = TRAINING_CONFIG["dataset"]["dataset_dir"]
            self.LABEL_PATH = TRAINING_CONFIG["dataset"]["label_path"]

            self.TRAINABLE_LAYERS = TRAINING_CONFIG["model"]["trainable_layers"]
            self.INITIALIZE_WEIGHT = TRAINING_CONFIG["model"]["initialize_weight"]

            self.EPOCHS = TRAINING_CONFIG["params"]["epochs"]

            self.EARLY_STOPPING_MONITOR = TRAINING_CONFIG["params"]["early_stopping_monitor"]
            self.EARLY_STOPPING_PATIENCE = TRAINING_CONFIG["params"]["early_stopping_patience"]
            self.EARLY_STOPPING_MODE = TRAINING_CONFIG["params"]["early_stopping_mode"]

            self.MODEL_CHECKPOINT_MONITOR = TRAINING_CONFIG["params"]["model_checkpoint_monitor"]
            self.MODEL_CHECKPOINT_MODE = TRAINING_CONFIG["params"]["model_checkpoint_mode"]


class EvalCoonfig:
      def __init__(self, config_path):
            print(config_path)
           
            with open(config_path) as configFile:
                  EVAL_CONFIG = json.load(configFile)

            self.COLORS = EVAL_CONFIG["mask_color"]
            
            for obj, color in self.COLORS.items():
                  self.COLORS[obj] = tuple(color)

            self.DEVICE = EVAL_CONFIG["device"]
            self.IMAGE_DIR = EVAL_CONFIG["image_dir"]
            self.LABEL_PATH = EVAL_CONFIG["label_path"]
            self.CLASSES_MAP_PATH = EVAL_CONFIG["classes_map_path"]
            self.TEST_SET = EVAL_CONFIG["test_set"]
            self.IOU_THRESHOLD = EVAL_CONFIG["iou_threshold"]
            self.MAX_BBOX_OVERLAP = EVAL_CONFIG["max_bbox_overlap"]