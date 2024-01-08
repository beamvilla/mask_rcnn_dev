import cv2
import numpy as np
from typing import Tuple, List

  
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


def polygon_to_binary_mask(
    all_points_x: List[int], 
    all_points_y: List[int],
    height: int,
    width: int
) -> np.array:
    area = []
    for i in range(len(all_points_x)):
        area.append([all_points_x[i], all_points_y[i]])
    area = np.array(area)
    mask = np.zeros((height, width))
    cv2.fillPoly(mask, [area], 1)
    mask = mask.astype(bool)
    return mask