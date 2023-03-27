import os
import cv2
import numpy as np
from skimage.measure import find_contours
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def apply_mask(image, mask, color, alpha=0.3):
    """
      Apply the given mask to the image.
      source: https://github.com/matterport/Mask_RCNN/blob/master/mrcnn/visualize.py
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image


def display_instances(image, boxes, masks, obj_names, colors,
                      save_pred_dir, image_file_name,
                      scores=None,
                      title="",
                      apply_box=True):
    """
    source code: https://www.programcreek.com/python/?CodeExample=display+instances
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [height, width, num_instances]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
    figsize: (optional) the size of the image.
    """
    _, ax = plt.subplots(1, figsize=(16, 16))

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis("off")
    ax.set_title(title)

    masked_image = image.astype(np.uint32).copy()
    for i in range(len(obj_names)):
        # Label
        score = str(round(scores[i], 2)) if scores is not None else None
        label = obj_names[i]

        if apply_box:
          # box
          y1, x1, y2, x2 = boxes[i]
          p = patches.Rectangle(
                                    (x1, y1), 
                                    x2 - x1, 
                                    y2 - y1, 
                                    linewidth=2,
                                    alpha=0.7, 
                                    linestyle="dashed",
                                    edgecolor=colors[label], 
                                    facecolor="none"
                                )
          ax.add_patch(p)

          ax.text(x1, y1, f"{label}: {score}",
                color='w', size=11, backgroundcolor="black")
          
        # Mask
        mask = masks[:, :, i] # Mask is in Boolean form (True, False)

        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        padded_mask = np.zeros(
                                (mask.shape[0] + 2, mask.shape[1] + 2), 
                                dtype=np.uint8
                              )
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5)
        for verts in contours:
            # Subtract the padding and flip (y, x) to (x, y)
            verts = np.fliplr(verts) - 1
            p = Polygon(verts, facecolor="none", edgecolor=colors[label])
            ax.add_patch(p)

    ax.imshow(masked_image.astype(np.uint8))
    plt.savefig(os.path.join(save_pred_dir, image_file_name),
                bbox_inches="tight", 
                pad_inches=-0.5,
                orientation="landscape")
    plt.show()


def visualize_gt_mask_on_image(gt_image, gt_mask, objects, save_pred_dir, image_file_name, colors, title="Ground-truth"):
    _, ax = plt.subplots(1, figsize=(16,16))

    height, width = gt_image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis("off")
    ax.set_title(title)

    masked_image = gt_image.astype(np.uint32).copy()

    for idx in range(len(objects)):
      mask = gt_mask[:, :, idx]
      # Mask Polygon
      # Pad to ensure proper polygons for masks that touch image edges.
      padded_mask = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
      padded_mask[1:-1, 1:-1] = mask
      contours = find_contours(padded_mask, 0.5)
      label = objects[idx]
      for verts in contours:
        # Subtract the padding and flip (y, x) to (x, y)
        verts = np.fliplr(verts) - 1
        print(f"color : {colors[label]}")
        p = Polygon(verts, facecolor="none", edgecolor=colors[label])
        ax.add_patch(p)
        
    ax.imshow(masked_image.astype(np.uint8))
    plt.savefig(os.path.join(save_pred_dir, image_file_name),
                bbox_inches="tight", 
                pad_inches=-0.5,
                orientation="landscape")
    plt.show() 