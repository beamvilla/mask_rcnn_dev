import json
import os
import argparse


# Initialize parser
parser = argparse.ArgumentParser()
 
# Adding optional argument
parser.add_argument("-project", "--project", help="Project name")
parser.add_argument("-subset", "--subset", help="Subset name")
parser.add_argument("-label_source_dir", "--label_source_dir", help="Label source directory.", default="./roseapple_new_dataset_label_16032023/")
parser.add_argument("-image_dir", "--image_dir", help="Image directory.", default="./roseapple_new_dataset_05122021")

# Read arguments from command line
args = parser.parse_args()

project = args.project
subset = args.subset
label_source_dir = args.label_source_dir
image_dir = args.image_dir


PATH = os.path.join(label_source_dir, project, f"{subset}_label.json")


with open(PATH) as f:
    labels = json.load(f)

via_format = {
    "_via_settings": {
        "ui": {
            "annotation_editor_height": 25,
            "annotation_editor_fontsize": 0.9,
            "leftsidebar_width": 18,
            "image_grid": {
                "img_height": 80,
                "rshape_fill": "none",
                "rshape_fill_opacity": 0.3,
                "rshape_stroke": "yellow",
                "rshape_stroke_width": 2,
                "show_region_shape": True,
                "show_image_policy": "all"
            },
            "image": {
                "region_label": "__via_region_id__",
                "region_color": "roseapple",
                "region_label_font": "10px Sans",
                "on_image_annotation_editor_placement": "NEAR_REGION"
            }
        },
        "core": {
            "buffer_size": 18,
            "filepath": {},
            "default_filepath": ""
        },
        "project": {
            "name": f"{project}_via"
        }
    },
    "_via_img_metadata": {},
    "_via_attributes": {
        "region": {
            "roseapple": {
                "type": "dropdown",
                "description": "",
                "options": {
                    "skin": "",
                    "minor": "",
                    "critical": ""
                },
                "default_options": {}
            }
        },
        "file": {}
    },
    "_via_data_format_version": "2.0.10",
    "_via_image_id_list": []
}

region_name = list(via_format["_via_attributes"]["region"].keys())[0]
for filename, attr in labels.items():
    regions = []
    image_path = os.path.join(image_dir, filename)
    image_size = os.path.getsize(image_path)
    image_key = filename + str(image_size)

    n_obj = len(attr["polygons"])

    for i in range(n_obj):
        regions.append(
            {
                "shape_attributes": {
                    "name": attr["polygons"][i]["name"],
                    "all_points_x": attr["polygons"][i]["all_points_x"],
                    "all_points_y": attr["polygons"][i]["all_points_y"],
                },
                "region_attributes": {
                    region_name: attr["objects"][i]
                }
            }
        )
    via_format["_via_img_metadata"][image_key] = {
        "filename": filename,
        "size": image_size,
        "regions": regions,
        "file_attributes": {}
    }

    via_format["_via_image_id_list"].append(image_key)


split_path = PATH.split("/")
dataset_dir = "/".join(split_path[:-1])
dataset_filename, dataset_format = split_path[-1].split(".")

output_obj = json.dumps(via_format, ensure_ascii=False)
with open(os.path.join(dataset_dir, dataset_filename + "_via_format." + dataset_format), "w") as f:
    f.write(output_obj)
