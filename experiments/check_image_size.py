from PIL import Image
import os
from tqdm import tqdm


IMAGE_PATH = "./classification_dataset/defect_type_dataset/"
SUBSETS = ["train", "val", "test"]
CLASSES = ["critical", "minor"]

width = []
height = []

for subset in tqdm(SUBSETS):
    for _class in CLASSES:
        image_dir = os.path.join(IMAGE_PATH, subset, _class)
        for image_filename in os.listdir(image_dir):
            image_path = os.path.join(image_dir, image_filename)
            image = Image.open(image_path)
            w, h = image.size
            width.append(w)
            height.append(h)

# Explore data analyse
max_width = max(width)
min_width = min(width)
mean_width = sum(width) / len(width)

max_height = max(height)
min_height = min(height)
mean_height = sum(height) / len(height)

print("-"*40)
print(" "*10 + "SUMMARY")
print("-"*40)
print(f"width >> max: {max_width}, min: {mean_width}, mean: {mean_width}")
print(f"height >> max: {max_height}, min: {mean_height}, mean: {mean_height}")