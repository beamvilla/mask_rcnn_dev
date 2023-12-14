import numpy as np

from .non_max_suppression import non_max_suppression


def filter(pred_results, max_bbox_overlap=0.75):
    skin_id = 1
    n_skin = 0

    for c in pred_results["class_ids"]:
        if c == skin_id:
            n_skin += 1

    h, w, c = pred_results["masks"].shape

    without_skin_rois = []
    without_skin_class_ids = []
    without_skin_scores = []
    without_skin_masks = np.zeros((h, w, c - n_skin))

    skin_rois = []
    skin_class_ids = []
    skin_scores = []
    skin_masks = np.zeros((h, w, n_skin))

    nxt_defect = 0
    nxt_skin = 0

    for i in range(len(pred_results["class_ids"])):
        if pred_results["class_ids"][i] != skin_id:
            without_skin_rois.append(pred_results["rois"][i])
            without_skin_class_ids.append(pred_results["class_ids"][i])
            without_skin_scores.append(pred_results["scores"][i])
            without_skin_masks[:, :, nxt_defect] = pred_results["masks"][:, :, i]
            nxt_defect += 1
        else:
            skin_rois.append(pred_results["rois"][i])
            skin_class_ids.append(pred_results["class_ids"][i])
            skin_scores.append(pred_results["scores"][i])
            skin_masks[:, :, nxt_skin] = pred_results["masks"][:, :, i]
            nxt_skin += 1

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
                    max_bbox_overlap=max_bbox_overlap,
                    scores=without_skin_scores
                )

    filtered_rois = []
    filtered_class_ids = []
    filtered_scores = []
    filtered_masks = np.zeros((h, w, c))

    nxt = 0
    for i in range(len(skin_rois)):
        filtered_rois.append(skin_rois[i])
        filtered_class_ids.append(skin_class_ids[i])
        filtered_scores.append(skin_scores[i])
        filtered_masks[:, :, nxt] = skin_masks[:, :, i]
        nxt += 1

    for idx in filtered_ids:
        filtered_rois.append(without_skin_rois[idx])
        filtered_class_ids.append(without_skin_class_ids[idx])
        filtered_scores.append(without_skin_scores[idx])
        filtered_masks[:, :, nxt] = without_skin_masks[:, :, idx]
        nxt += 1

    filtered_pred_results = {
            "rois": np.array(filtered_rois),
            "class_ids": np.array(filtered_class_ids),
            "scores": np.array(filtered_scores),
            "masks": np.array(filtered_masks)
        }
    return filtered_pred_results