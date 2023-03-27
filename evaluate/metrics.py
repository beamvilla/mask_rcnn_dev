import numpy as np
import pandas as pd


def overlap_masks(pred_results, gt_masks):
    # Change to results from trim_masks
    pred_scores = pred_results["scores"]

    # Sort predictions by score from high to low
    indices = np.argsort(pred_scores)[::-1]
    pred_scores = pred_scores[indices]

    # Sort predict masks
    pred_masks = pred_results["masks"]
    pred_masks = pred_masks[..., indices]

    print(f"[PREDICT MASK SHAPE]: {pred_masks.shape}")
    print(f"[GT MASK SHAPE]: {gt_masks.shape}")

    if pred_masks.shape[-1] == 0:
       return 0
    
    # Reshape masks
    reshaped_pred_masks = np.reshape(pred_masks > .5, (-1, pred_masks.shape[-1])).astype(np.float32)
    reshaped_gt_masks = np.reshape(gt_masks > .5, (-1, gt_masks.shape[-1])).astype(np.float32)

    # Calculate masks area
    area_pred_masks = np.sum(reshaped_pred_masks, axis=0)
    area_gt_masks = np.sum(reshaped_gt_masks, axis=0)

    # Intersection masks
    masks_intersections = np.dot(reshaped_pred_masks.T, reshaped_gt_masks)

    # Union masks
    union_masks = area_pred_masks[:, None] + area_gt_masks[None, :] - masks_intersections

    # Mask overlap
    masks_overlaps = masks_intersections / union_masks
    return masks_overlaps


def matching_class(pred_results, gt_masks, pred_class_ids, gt_class_ids,
                   score_threshold=0.0, iou_threshold=0.5):
    wrong_class_val_dict =  {
                                (2, 3): -2,
                                (3, 2): -3,
                                (2, 1): -4,
                                (3, 1): -5,
                                (1, 2): -6,
                                (1, 3): -7
                            }
    masks_overlaps = overlap_masks(pred_results, gt_masks)
    pred_match = -1 * np.ones([len(pred_class_ids)])
    gt_match = -1 * np.ones([len(gt_class_ids)])
    
    for i in range(len(pred_class_ids)):
      # Find best matching ground truth box
      # 1. Sort matches by score (high to low)
      sorted_ixs = np.argsort(masks_overlaps[i])[::-1]  # return index array, the array columns equal ground truth length   
      # 2. Remove low scores
      low_score_idx = np.where(masks_overlaps[i, sorted_ixs] < score_threshold)[0]
      if low_score_idx.size > 0:
          sorted_ixs = sorted_ixs[:low_score_idx[0]]
      # 3. Find the match
      for j in sorted_ixs:
          # If ground truth box is already matched, go to next one
          if gt_match[j] > -1:
              continue
          
          # If we reach IoU smaller than the threshold, end the loop
          iou = masks_overlaps[i, j]
          if iou < iou_threshold:
            break
          
          # Do we have a match?
          # i are represented prediction, j are represented ground truth
          if pred_class_ids[i] == gt_class_ids[j]:
              gt_match[j] = i   # i -->  predicted class index
              pred_match[i] = j  # j -->  ground truth class index
              break
          
          # In case of IoU > threshold but predict wrong class
          else:
            _wrong_match_val = wrong_class_val_dict[(gt_class_ids[j], pred_class_ids[i])]
            gt_match[j] = _wrong_match_val
            pred_match[i] = _wrong_match_val
            break
    print(f"gt_match: {gt_match} pred_match: {pred_match} \n")
    return gt_match, pred_match


def cal_confusion_matrix(confusion_scores_dict, pred_results, gt_masks, pred_class_ids, gt_class_ids,
                         score_threshold=0.0, iou_threshold=0.5):
  
    # Calculate gt_match and pred_match
    gt_match, pred_match = matching_class(pred_results,
                                          gt_masks,
                                          pred_class_ids,
                                          gt_class_ids,
                                          score_threshold=score_threshold, 
                                          iou_threshold= iou_threshold)
    
    # Extract value which more than -1 to find class matching
    gt_match_idx = sorted([gt.astype("int") for gt in gt_match if gt > -1])
    pred_match_idx = [pred.astype("int") for pred in pred_match if pred > -1]
    
    # Create list for checking which index was used
    check_pred_used = -1 * np.ones([len(pred_match_idx)])

    # Loop throught match class
    for gt_id in gt_match_idx:
      for num,pred_id in enumerate(pred_match_idx):
        if check_pred_used[num] != -1:
          continue
        try:
          confusion_scores_dict[gt_class_ids[pred_id] - 1, pred_class_ids[gt_id]-1] += 1
        except KeyError:
          pass
        else:
          check_pred_used[num] = 0
          break

    # In case of IoU > threshold but predict wrong class
    wrong_match_scores = {
                              -2: (2, 1), 
                              -3: (1, 2), 
                              -4: (0, 1), 
                              -5: (0, 2), 
                              -6: (1, 0), 
                              -7: (2, 0)
                          }
    
    for k in wrong_match_scores:
      wrong_match_size = len([gt[0] for gt in enumerate(gt_match) if gt[1] == k])
      if wrong_match_size > 0:
        confusion_scores_dict[wrong_match_scores[k]] += wrong_match_size
    
    # Find index of -1 for finding incorrect prediction
    gt_non_match_idx = [gt[0] for gt in enumerate(gt_match) if gt[1] == -1]
    pred_non_match_idx = [pred[0] for pred in enumerate(pred_match) if pred[1] == -1]

    # False Negative case
    FN_non_match_score = {
                            1: (4, 0), 
                            2: (3, 1), 
                            3: (3, 2)
                         }
    
    # False Positive case
    FP_non_match_score = {
                            1: (0, 4), 
                            2: (1, 3), 
                            3: (2, 3)
                         }
    
    # Loop through ground-truth label
    if len(gt_non_match_idx) > 0:
      for gt_idx in gt_non_match_idx:
        _idx = FN_non_match_score[gt_class_ids[gt_idx]]
        confusion_scores_dict[_idx] += 1
      
    # Loop through predicted label
    if len(pred_non_match_idx) > 0:
      for pred_idx in pred_non_match_idx:
        _idx = FP_non_match_score[pred_class_ids[pred_idx]]
        confusion_scores_dict[_idx] += 1

    return confusion_scores_dict


def cal_recall(confusion_df, class_name):
  sum_row = sum(confusion_df.loc[:, class_name].dropna())
  if sum_row == 0:
     return 0
  return confusion_df[class_name][class_name] / sum_row


def cal_precision(confusion_df, class_name):
  sum_row = sum(confusion_df.loc[class_name, :].dropna())
  if sum_row == 0:
     return 0
  return confusion_df[class_name][class_name] / sum_row


# Create confusion-matrix dataframe
def export_metric_result(confusion_scores_dict, metrics_list, classes_name):
    confusion_df = pd.DataFrame(index=metrics_list, columns=metrics_list)

    for idx in confusion_scores_dict:
      confusion_df.iloc[idx[0], idx[1]] = confusion_scores_dict[idx]

    recall_and_precision_metric = pd.DataFrame(index=classes_name,
                                               columns=["recall", "precision"])
    
    # Recall
    for c in classes_name:
        recall_and_precision_metric["recall"][c] = cal_recall(confusion_df, c)
        recall_and_precision_metric["precision"][c] = cal_precision(confusion_df, c)

    return confusion_df, recall_and_precision_metric