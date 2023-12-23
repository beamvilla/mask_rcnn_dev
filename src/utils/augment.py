import numpy as np
import PIL
from PIL import Image, ImageEnhance
from typing import List, Tuple


def flip_polygon_vertical(
    image_width: int,
    x_polygon: List[int]
) -> List[int]:
    # image_size: width
    all_points_x = np.array(x_polygon)
    all_points_x_flipped = image_width - all_points_x
    return all_points_x_flipped.tolist()


def augment_flip_image_vertical(
    image: Image.Image, 
    polygons: List[int], 
    image_size: int
) -> Tuple[Image.Image, List[int]]:
    augmented_image = image.transpose(Image.FLIP_LEFT_RIGHT)
    polygons = flip_polygon_vertical(
                    image_width=image_size,
                    x_polygon=polygons
                )
    return augmented_image, polygons


def flip_polygon_horizontal(
    image_height: int, 
    y_polygon: List[int]
) -> List[int]:
    # image_size: height
    all_points_y = np.array(y_polygon)
    all_points_y_flipped = image_height - all_points_y
    return all_points_y_flipped.tolist()


def augment_flip_image_horizontal(
    image: Image.Image, 
    polygons: List[int], 
    image_size: int
) -> Tuple[Image.Image, List[int]]:
    augmented_image = image.transpose(Image.FLIP_TOP_BOTTOM)
    polygons = flip_polygon_horizontal(
                    image_height=image_size,
                    y_polygon=polygons
                )
    return augmented_image, polygons


def augment_brightness(
    image: Image.Image, 
    volume: float
):
    image = ImageEnhance.Brightness(image)
    augmented_image = image.enhance(volume)
    return augmented_image