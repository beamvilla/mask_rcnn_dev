import sys
sys.path.append("..")

import os
import json
import skimage.draw
import numpy as np
from typing import Dict

from mrcnn import utils


class CustomDataset(utils.Dataset):
    def load_custom(
        self, 
        dataset_dir: str, 
        annotation_path: str, 
        classes_map: Dict[str, int]
    ) -> None:
        cnt_classes = {}
        for class_name, class_id in classes_map.items():
            cnt_classes[class_name] = 0
            self.add_class("object", class_id, class_name)
        
        with open(annotation_path) as annoFile:
            annotations = json.load(annoFile)

        img_metadata = annotations["_via_img_metadata"]

        # Add images
        for _, metadata in img_metadata.items():
            num_ids = []
            polygons = []
            image_file_name = metadata["filename"]
            regions = metadata["regions"]

            for region in regions:
                polygons.append(region["shape_attributes"])
                for region_attr in region["region_attributes"].values():
                    cnt_classes[region_attr] += 1
                    num_ids.append(classes_map[region_attr])

            image_path = os.path.join(dataset_dir, image_file_name)
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "object",
                image_id=image_file_name,  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons,
                num_ids=num_ids
            )
            
        print(cnt_classes)

    def load_mask(self, image_id):
        """
        Generate instance masks for an image.
        Returns:
        masks: A bool array of shape [height, width, instance count] with
                one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a Horse/Man dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "object":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        if info["source"] != "object":
            return super(self.__class__, self).load_mask(image_id)
        
        num_ids = info["num_ids"]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)

        # Use multiple shapes i.e. circle, ellipse, and polygon
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            if p["name"] == "polygon":
                rr, cc = skimage.draw.polygon(p["all_points_y"], p["all_points_x"])            
            elif p["name"] == "circle":
                rr, cc = skimage.draw.circle(p["cx"], p["cy"], p["r"])
            else: 
                rr, cc = skimage.draw.ellipse(p["cx"], p["cy"], p["rx"], p["ry"], rotation=np.deg2rad(p["theta"]))  
            rr[rr > mask.shape[0]-1] = mask.shape[0] - 1
            cc[cc > mask.shape[1]-1] = mask.shape[1] - 1
            mask[rr, cc, i] = 1

        num_ids = np.array(num_ids, dtype=np.int32)
        return mask, num_ids

    def image_reference(self, image_id):
        """
        Return the path of the image.
        """
        info = self.image_info[image_id]
        if info["source"] == "object":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)