import cv2
import numpy as np

  
def plot_masks(masks: np.array) -> np.array:
    for mask in masks:
        isClosed = True
        color = (255, 0, 0)
        thickness = 2
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