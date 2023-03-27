import os
import matplotlib.pyplot as plt
import sys
sys.path.append("..")

from utils.visualize import display_instances, visualize_gt_mask_on_image
from evaluate.ground_truth import extract_anno_gt
from evaluate.metrics import cal_confusion_matrix


def evaluate(image_dir, annotations, confusion_scores_dict, model, classes_map, colors, save_pred_dir, iou_threshold=0.5):
    classes_map_id = {}
    for class_name, class_id in classes_map.items():
        classes_map_id[class_id] = class_name

    for image_file_name, anno_data in annotations.items():
        print(f"[FILENAME]: {image_file_name}")

        image_path = os.path.join(image_dir, image_file_name)
        pred_obj_names = []

        # Import ground-truth image
        gt_polygons = anno_data["polygons"]
        gt_objects = anno_data["objects"]
        gt_image, gt_masks, gt_label_ids = extract_anno_gt(
                                                            image_path=image_path, 
                                                            gt_polygons=gt_polygons, 
                                                            gt_objects=gt_objects,
                                                            classes_map=classes_map
                                                        )
        
        # Run object detection
        pred_results = model.detect([gt_image], verbose=1)
        pred_results = pred_results[0]
      
        for id in pred_results["class_ids"]:
            pred_obj_names.append(classes_map_id[id])

        # Visualize ground-truth
        display_instances(
                            image=gt_image, 
                            boxes=None, 
                            masks=gt_masks, 
                            obj_names=gt_objects, 
                            colors=colors,
                            title="Ground-Truth",
                            scores=None, 
                            apply_box=False,
                            save_pred_dir=save_pred_dir,
                            image_file_name=image_file_name.split(".")[0] + "_gt.png"
                        )
        """
        visualize_gt_mask_on_image(
            gt_image=gt_image,
            gt_mask=gt_masks,
            objects=gt_objects,
            save_pred_dir=save_pred_dir,
            colors=colors,
            image_file_name=image_file_name.split(".")[0] + "_gt.png"
        )
        """
        # Visualize prediction
        display_instances(
                            image=gt_image, 
                            boxes=pred_results["rois"], 
                            masks=pred_results["masks"], 
                            obj_names=pred_obj_names, 
                            colors=colors,
                            title="prediction",
                            scores=pred_results["scores"], 
                            save_pred_dir=save_pred_dir,
                            image_file_name=image_file_name.split(".")[0] + "_pred.png"
                        )

        _ = cal_confusion_matrix(
                                    confusion_scores_dict=confusion_scores_dict, 
                                    pred_results=pred_results,
                                    gt_masks=gt_masks, 
                                    pred_class_ids=pred_results["class_ids"],
                                    gt_class_ids=gt_label_ids,
                                    iou_threshold=iou_threshold
                                )
    return confusion_scores_dict