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


def union(a,b):
  x1 = min(a[0], b[0])
  y1 = min(a[1], b[1])
  x2 = max(a[2], b[2])
  y2 = max(a[3], b[3])
  return [x1, y1, x2, y2]


def area(box1, box2):  # returns None if rectangles don't intersect
    dx = min(box1[2], box2[2]) - max(box1[0], box2[0])
    dy = min(box1[3], box2[3]) - max(box1[1], box2[1])

    if (dx>=0) and (dy>=0):
        return dx * dy
    return 0.0


def combine_boxes(boxes, overlap_threshold=0.25):
    if len(boxes) <= 1:
        return boxes

    id = 0

    while id < len(boxes):
        combined = False
        for j in range(id + 1, len(boxes)):
            if area(boxes[id], boxes[j]) > overlap_threshold:
                combined = True
                boxes[id] = union(boxes[id], boxes[j])
                boxes.remove(boxes[j])
                break
        if not combined:
            id += 1
    return boxes


def get_iou(box1: List[int], box2: List[int]) -> float:
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    box1Area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2Area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(box1Area + box2Area - interArea)
    # return the intersection over union value
    return iou


def not_in_area(
    specific_box: List[int], 
    object_box: List[int]
  ) -> bool:
    """
    Check that object is in specific area.

    Parameters
    ----------
    target_box : list
        [x1, x2, y1, y2]
    check_box : list
        [x1, x2, y1, y2]

    Returns
    -------
    bool
    """
    is_not_in_area = False
    iou = get_iou(box1=object_box, box2=specific_box)
    
    if iou <= 0.0:
       is_not_in_area = True
    return is_not_in_area