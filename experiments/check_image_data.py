import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import List


def check_duplicate_train_image(
    main_images_dir: str, 
    another_images_dir: str 
) -> List[str]:
    duplicate_images_path = []

    for main_image_filename in tqdm(os.listdir(main_images_dir)):
        main_image_path = os.path.join(main_images_dir, main_image_filename)
        main_image_array = np.array(Image.open(main_image_path))

        for another_image_filename in os.listdir(another_images_dir):
            another_image_path = os.path.join(another_images_dir, another_image_filename)
            another_image_array = np.array(Image.open(another_image_path))

            if np.array_equal(main_image_array, another_image_array):
                print("Found: " + another_image_path)
                duplicate_images_path.append(another_image_path)
    
    return duplicate_images_path



duplicate_images_path = check_duplicate_train_image(
    main_images_dir="./dataset/white_bg/images/train",
    another_images_dir="./dataset/white_bg/images/test"
)