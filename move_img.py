from PIL import Image
import os
import json


PATH = "./roseapple_new_dataset_label_16032023/test/test_label.json"
IMAGE_DIR = "./roseapple_new_dataset_05122021/"
image_des = os.path.join(IMAGE_DIR, "test")

if not os.path.exists(image_des):
    os.makedirs(image_des)

with open(PATH) as f:
    labels = json.load(f)

for filename, _ in labels.items():
    image = Image.open(os.path.join(IMAGE_DIR, filename))
    image.save(os.path.join(image_des, filename))