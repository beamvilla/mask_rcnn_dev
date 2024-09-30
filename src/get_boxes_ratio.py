def get_boxes_ratio(boxes):
    ratios = []
    for box in boxes:
        w, h = box
        r = h / w
        ratios.append(r)
    return ratios

boxes = [[37, 36], [211, 229], [43, 45], [50, 49], [58, 51], [40, 41], [67, 71], [43, 44], [46, 44], [48, 56], [58, 59], [59, 63], [51, 58], [65, 61], [41, 42]]


ratios = get_boxes_ratio(boxes)
print(ratios)