import os
import sys
sys.path.append("..")

#from utils.visualize import display_instances
from utils.filter_overlap_defect import filter
from evaluate.ground_truth import extract_anno_gt
from evaluate.metrics import cal_confusion_matrix


def evaluate(
    image_dir, 
    annotations, 
    confusion_scores_dict, 
    model, 
    classes_map, 
    colors, 
    save_pred_dir, 
    iou_threshold=0.5, 
    max_bbox_overlap=0.75
):
    classes_map_id = {}
    for class_name, class_id in classes_map.items():
        classes_map_id[class_id] = class_name

    img_metadata = annotations["_via_img_metadata"]
    for _, metadata in img_metadata.items():
        gt_polygons = []
        gt_objects = []

        image_file_name = metadata["filename"]
        regions = metadata["regions"]

        print("[FILENAME]: " + image_file_name)

        image_path = os.path.join(image_dir, image_file_name)
        pred_obj_names = []

        # Import ground-truth image
        for region in regions:
            gt_polygons.append(region["shape_attributes"])
            for region_attr in region["region_attributes"].values():
                gt_objects.append(region_attr)

        gt_image, gt_masks, gt_label_ids = extract_anno_gt(
                                                            image_path=image_path, 
                                                            gt_polygons=gt_polygons, 
                                                            gt_objects=gt_objects,
                                                            classes_map=classes_map
                                                        )
        
        # Run object detection
        pred_results = model.detect([gt_image], verbose=1)
        pred_results = pred_results[0]
        pred_results = filter(pred_results, max_bbox_overlap)

        for id in pred_results["class_ids"]:
            pred_obj_names.append(classes_map_id[id])

        # Visualize ground-truth
        """
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
        # Visualize prediction
        """
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
        """
        _ = cal_confusion_matrix(
                                    confusion_scores_dict=confusion_scores_dict, 
                                    pred_results=pred_results,
                                    gt_masks=gt_masks, 
                                    pred_class_ids=pred_results["class_ids"],
                                    gt_class_ids=gt_label_ids,
                                    iou_threshold=iou_threshold
                                )
    return confusion_scores_dict