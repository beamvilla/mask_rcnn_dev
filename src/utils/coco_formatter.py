import json
from typing import Dict, List
from datetime import datetime


class COCOAnnotation:
    def __init__(
        self,
        export_path: str,
        classes_maps: Dict[int, str]
    ) -> None:
        now = datetime.now()

        self.created_year = now.year
        self.created_time = now.strftime("%Y-%m-%dT%H:%M:%S")

        self.export_path = export_path
        self.classes_maps = classes_maps

        # Create an empty list to store the annotations
        self.annotations = []
        self.image_infos = []
        self.categories = []

        self.anno_ID = 1
        self.image_ID = 1

    def update_image_info(
        self,
        image_width: int,
        image_height: int,
        image_filename: str
    ) -> None:
        image_info = {
                        "id": self.image_ID,  # Use the same identifier as the annotation
                        "width": image_width,  # Set the width of the image
                        "height": image_height,  # Set the height of the image
                        "file_name": image_filename,  # Set the file name of the image
                    }
        self.image_infos.append(image_info)

    def update_detection_items(
        self,
        class_ids: List[int],
        segmentations: List[List[int]],
        boxes: List[List[int]]
    ) -> None:
        for i in range(len(class_ids)):
            # Annotate the image with a bounding box and label
            annotation = {
                "id": self.anno_ID,  # Use a unique identifier for the annotation
                "image_id": self.image_ID,  # Use the same identifier for the image
                "category_id": class_ids[i] + 1,  # Assign a category ID to the object
                "segmentation": [segmentations[i]],
                "bbox": boxes[i],  # Specify the bounding box in the format [x, y, width, height]
                "area": boxes[i][2] * boxes[i][3],  # Calculate the area of the bounding box
                "iscrowd": 0,  # Set iscrowd to 0 to indicate that the object is not part of a crowd
            }

            self.annotations.append(annotation)
            self.categories.append({"id": class_ids[i] + 1, "name": self.classes_maps[class_ids[i]]})
            self.anno_ID += 1
        self.image_ID += 1

    def export(self):
        # Create the COCO JSON object
        coco_data = {
            "info": {
                "description": "My COCO dataset",  # Add a description for the dataset
                "url": "",  # Add a URL for the dataset (optional)
                "version": "1.0",  # Set the version of the dataset
                "year": self.created_year,  # Set the year the dataset was created
                "contributor": "",  # Add the name of the contributor (optional)
                "date_created": self.created_time,  # Set the date the dataset was created
            },
            "licenses": [],  # Add a list of licenses for the images in the dataset (optional)
            "images": self.image_infos,
            "annotations": self.annotations,  # Add the list of annotations to the JSON object
            "categories": self.categories,  # Add a list of categories for the objects in the dataset
        }

        # Save the COCO JSON object to a file
        with open(self.export_path, "w") as f:
            json.dump(coco_data, f)