import cv2
import numpy as np
from typing import Tuple, List


def scale_bbox(
    bbox: Tuple[int, int, int, int], 
    original_shape: Tuple[int, int], 
    resized_shape: Tuple[int, int]
) -> List[int]:
    original_height, original_width = original_shape
    resized_height, resized_width = resized_shape

    Wratio = resized_width / original_width
    Hratio = resized_height / original_height

    ratioList = [Hratio, Wratio, Hratio, Wratio]
    bbox = [int(a * b) for a, b in zip(bbox, ratioList)]
    return bbox


def get_redefined_bbox(
    image: np.array, 
    bbox: Tuple[int, int, int, int], 
    base: int =512
) -> List[int]:
    original_height, original_width, _ = image.shape
    if original_height > original_width:
        height_persentage = float(base / original_height)
        width_size = int(original_width * height_persentage)
        resized_image = cv2.resize(image, (width_size, base), interpolation=cv2.INTER_CUBIC)
        resized_height, resized_width, _ = resized_image.shape
        bbox = scale_bbox(
                bbox=bbox, 
                original_shape=(original_height, original_width), 
                resized_shape=(resized_height, resized_width)
            )
        width1 = (base - resized_width) // 2
        width2 = (base - resized_width) - width1
        bbox = [bbox[0] + width1, bbox[1], bbox[2] + width2, bbox[3]]
        
    else:
        width_percentage = float(base / original_width)
        height_size = int(original_height * width_percentage)
        resized_image = cv2.resize(image, (base, height_size), interpolation=cv2.INTER_CUBIC)
        resized_height, resized_width, _ = resized_image.shape
        bbox = scale_bbox(
                bbox=bbox, 
                original_shape=(original_height, original_width), 
                resized_shape=(resized_height, resized_width)
            )
        height1 = (base - resized_height) // 2
        height2 = (base - resized_height) - height1
        bbox = [bbox[0], bbox[1] + height1, bbox[2], bbox[3] + height2]
    
    return bbox