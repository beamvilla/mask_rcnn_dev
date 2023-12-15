from typing import List, Tuple


def polygon_to_rect(
    polygon_x: List[int], 
    polygon_y: List[int]
) -> Tuple[int, int, int, int]:
    x = min(polygon_x)
    y = min(polygon_y)

    w = max(polygon_x) - x
    h = max(polygon_y) - y
    return x, y, w, h


def scale_bbox_to_xywh(
    bbox: List[List[float]],
    image_width: int,
    image_height: int
) -> List[List[int]]:
    scaled_bbox = []
    for box in bbox:
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]

        x = int((x - 0.5 * w) * image_width)
        y = int((y - 0.5 * h) * image_height)
        w = int(w * image_width)
        h = int(h * image_height)
        scaled_bbox.append([x, y, w, h])
    return scaled_bbox