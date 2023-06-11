import numpy as np

from utils.non_max_suppression import non_max_suppression


def filter(pred_results):
    filtered_rois = []
    filtered_class_ids = []
    filtered_scores = []
    filtered_masks = []

    without_skin_rois = []
    without_skin_class_ids = []
    without_skin_scores = []
    without_skin_masks = []

    skin_rois = []
    skin_class_ids = []
    skin_scores = []
    skin_masks = []

    skin_id = 1
    for i, c in enumerate(pred_results["class_ids"]):
        if c != skin_id:
            without_skin_rois.append(pred_results["rois"][i])
            without_skin_class_ids.append(pred_results["class_ids"][i])
            without_skin_scores.append(pred_results["scores"][i])
            without_skin_masks.append(pred_results["masks"][:, :, i])
        else:
            skin_rois.append(pred_results["rois"][i])
            skin_class_ids.append(pred_results["class_ids"][i])
            skin_scores.append(pred_results["scores"][i])
            skin_masks.append(pred_results["masks"][:, :, i])

    without_skin_rois = np.array(without_skin_rois)
    without_skin_class_ids = np.array(without_skin_class_ids)
    without_skin_scores = np.array(without_skin_scores)
    without_skin_masks = np.array(without_skin_masks)        

    skin_rois = np.array(skin_rois)
    skin_class_ids = np.array(skin_class_ids)
    skin_scores = np.array(skin_scores)
    skin_masks = np.array(skin_masks)

    filtered_ids = non_max_suppression(
                    boxes=without_skin_rois,
                    max_bbox_overlap=0.75,
                    scores=without_skin_scores
                )
    
    for idx in filtered_ids:
        filtered_rois.append(without_skin_rois[idx])
        filtered_class_ids.append(without_skin_class_ids[idx])
        filtered_scores.append(without_skin_scores[idx])
        filtered_masks.append(without_skin_masks[:, :, idx])
        
    filtered_rois.extend(skin_rois)
    filtered_class_ids.extend(skin_class_ids)
    filtered_scores.extend(skin_scores)
    filtered_masks.extend(skin_masks)

    filtered_masks = np.moveaxis(filtered_masks, -1, 0)
    filtered_pred_results = {
        "rois": np.array(filtered_rois),
        "class_ids": np.array(filtered_class_ids),
        "scores": np.array(filtered_scores),
        "masks": np.array(filtered_masks)
    }
    return filtered_pred_results