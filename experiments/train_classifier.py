import os
import numpy as np
from typing import List, Tuple, Dict
from PIL import Image
from tqdm import tqdm
from datetime import datetime
import tensorflow as tf

# import the necessary packages
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import MaxPooling2D, AveragePooling2D
from tensorflow.keras.models import Sequential


def prepare_dataset(
    dataset_dir: str,
    resize: int = 128
) -> Tuple[np.array, np.array]:
    labels = []
    images = []
    for classname in tqdm(os.listdir(dataset_dir)):
        class_id = classes_map[classname]
        image_dir = os.path.join(dataset_dir, classname)
        for image_filename in os.listdir(image_dir):
            labels.append(class_id)
            image_path = os.path.join(image_dir, image_filename)
            image = Image.open(image_path)
            image = image.resize((resize, resize))
            image = np.array(image)
            image = image.astype(float)
            image /= 255.0
            images.append(image)
    return np.array(images), np.array(labels)


def build_model():
    model = Sequential()
    model.add(Conv2D(128, (3, 3), activation="relu", input_shape=(128, 128, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(256, activation="relu"))
    model.add(Dense(len(np.unique(train_y)), activation="sigmoid"))
    return model


class SaveBestModel(tf.keras.callbacks.Callback):
    def __init__(self, model_path, metric):
        self.model_path = model_path
        self.metric = metric


    def on_epoch_end(self, epoch, logs=None):
        if epoch == 0 or logs[self.metric] < self.minimum_loss:
            self.minimum_loss = logs[self.metric]
            model.save(self.model_path)
            print("Saved best model")


now = datetime.today().strftime("%Y_%m_%d_%H_%M_%S")
model_path = f"./defect_classification/{now}"

if not os.path.exists(model_path):
  os.makedirs(model_path)


DATASET_DIR = "./classification_dataset/defect_type_dataset"
SUBSET = ["train", "val", "test"]
classes_map = {
    "minor": 0,
    "critical": 1
}

train_x, train_y = prepare_dataset(os.path.join(DATASET_DIR, "train"))
val_x, val_y = prepare_dataset(os.path.join(DATASET_DIR, "val"))
test_x, test_y = prepare_dataset(os.path.join(DATASET_DIR, "test"))

train_y = tf.keras.utils.to_categorical(train_y)
val_y = tf.keras.utils.to_categorical(val_y)
test_y = tf.keras.utils.to_categorical(test_y)

epochs = 100
batch_size = 32
step_per_epoch = len(train_x) // batch_size
metric = "val_loss"
patience = 10


early_stop=tf.keras.callbacks.EarlyStopping(monitor=metric, restore_best_weights=True, patience=patience, verbose=1)
callback = [early_stop]
callback.append(SaveBestModel(model_path, metric))


model = build_model()
model.compile(optimizer="adam",
              loss="binary_crossentropy",
              metrics=["accuracy"])

history = model.fit(
    train_x, train_y,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(val_x, val_y),
    callbacks=callback
  )

test_loss, test_acc = model.evaluate(test_x, test_y)
print('Test accuracy:', test_acc)