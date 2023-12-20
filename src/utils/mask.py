import cv2
import numpy as np
from typing import Tuple

  
def plot_masks(
    masks: np.array, 
    image: np.array,
    color: Tuple[int, int, int] = (255, 0, 0),
    isClosed: bool = True,
    thickness: int = 1
) -> np.array:
    for mask in masks:
        image = cv2.polylines(
                    image, 
                    np.int32([mask]), 
                    isClosed, 
                    color, 
                    thickness
                )
    return image


def rescale_mask(
    masks: np.array, 
    image_width: int, 
    image_height: int
) -> np.array:
    for mask in masks:
        mask[:, 0] = mask[:, 0] * image_width
        mask[:, 1] = mask[:, 1] * image_height
        mask.astype(int)
    return masks