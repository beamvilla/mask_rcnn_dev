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