# anchor_w = [32, 64, 128, 256, 512]
# aspect_ratio = [0.5, 1, 2]

# all_anchors = []
# for w in anchor_w:
#     for a in aspect_ratio:
#         all_anchors.append([w, int(w * a)])


predict_anchors = [[38, 36], [51, 57], [58, 65], [42, 45], [41, 38], [47, 45], [34, 38], [55, 50], [55, 55], [211, 229], [51, 53], [64, 63], [42, 46], [62, 63], [63, 68]]

aspect_ratio = set()
for anchor in predict_anchors:
    w, h = anchor
    aspect_ratio.add(h / w)
print(list(aspect_ratio))
