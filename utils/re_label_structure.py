import json


DATASET_TYPE = "test"
LABEL_FILE_PATH = f"./roseapple_new_dataset_label_16032023/{DATASET_TYPE}/{DATASET_TYPE}_set_fixed_via_format.json"
RE_FORMAT_LABEL_FILE_PATH = f"./roseapple_new_dataset_label_16032023/{DATASET_TYPE}/{DATASET_TYPE}_label.json"

with open(LABEL_FILE_PATH) as labelFile:
    label_file = json.load(labelFile)

new_json_label_map = {}

region_attributes_key = list(label_file["_via_attributes"]["region"].keys())[0]

for attrs in label_file["_via_img_metadata"].values():
    new_objects_map = { 
                        "polygons" : [],
                        "objects" : []
                    }

    for region in attrs["regions"]:
        new_objects_map["polygons"].append(region["shape_attributes"])
        new_objects_map["objects"].append(region["region_attributes"][region_attributes_key])
    

    new_json_label_map[attrs["filename"]] = new_objects_map


print(len(new_json_label_map))

with open(RE_FORMAT_LABEL_FILE_PATH, "w") as labelFile:
    json.dump(new_json_label_map, labelFile)
