import json


DATASET = "test"
with open(f"./roseapple_new_dataset_label_16032023/{DATASET}/{DATASET}_label.json") as jsonFile:
    label = json.load(jsonFile)

max_num = -1

for l in label:
    num = len(label[l]["objects"])
    if num > max_num:
        max_num = num

print(max_num)
