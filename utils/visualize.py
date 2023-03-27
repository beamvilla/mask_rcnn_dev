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
    # Number of instances
    N = boxes.shape[0]

    _, ax = plt.subplots(1, figsize=(16, 16))

    # Generate specific colors
    # skin clor --> red
    skin_color = (1.0, 0.03, 0.00)
    minor_defect_color = (0.21, 0.77, 0.74)
    critical_defect_color = (0.99, 0.89, 0.02)


    # Show area outside image boundaries.
    height, width = image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')
    ax.set_title(title)

    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        #color = colors[i]

        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        # Label
        score = scores[i] if scores is not None else None
        label = obj_names[i]

        #box
        y1, x1, y2, x2 = boxes[i]

        if apply_box:
          if label == "skin":
            #red
            p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                  alpha=0.7, linestyle="dashed",
                                  edgecolor=skin_color, facecolor='none')
            ax.add_patch(p)
          elif label == "minor":
            #yellow
            p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                  alpha=0.7, linestyle="dashed",
                                  edgecolor=minor_defect_color, facecolor='none')
            ax.add_patch(p) 
          elif label == "critical":
            #yellow
            p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                  alpha=0.7, linestyle="dashed",
                                  edgecolor=critical_defect_color, facecolor='none')
            ax.add_patch(p) 

        ax.text(x1, y1, f"{label}: {round(score, 2)}",
                color='w', size=11, backgroundcolor="black")

        # Mask
        mask = masks[:, :, i] # Mask is in Boolean form (True, False)

        if apply_mask:
          if label == "skin":
            # red
            masked_image = apply_mask(masked_image, mask, skin_color)
          elif label == "minor":
            # blue
            masked_image = apply_mask(masked_image, mask, minor_defect_color)
          elif label == "critical":
            # yellow
            masked_image = apply_mask(masked_image, mask, critical_defect_color)

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
            if label == "skin":
              #red
              p = Polygon(verts, facecolor="none", edgecolor=skin_color)
              ax.add_patch(p)
            elif label == "minor":
              #blue
              p = Polygon(verts, facecolor="none", edgecolor=minor_defect_color)
              ax.add_patch(p)
            elif label == "critical":
              #yellow
              p = Polygon(verts, facecolor="none", edgecolor=critical_defect_color)
              ax.add_patch(p)

    ax.imshow(masked_image.astype(np.uint8))
    plt.savefig(os.path.join(save_pred_dir, image_file_name),
                bbox_inches='tight', 
                pad_inches=-0.5,
                orientation= 'landscape')
    plt.show()


def visualize_gt_mask_on_image(gt_image, gt_mask, objects, save_pred_dir, image_file_name, title="Ground-truth", show_mask=0):
    skin_color = (1.0, 0.03, 0.00)
    minor_defect_color = (0.21, 0.77, 0.74)
    critical_defect_color = (0.99, 0.89, 0.02)

    _, ax = plt.subplots(1, figsize=(16,16))

    height, width = gt_image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')
    ax.set_title(title)

    masked_image = gt_image.astype(np.uint32).copy()

    for idx in range(len(objects)):
      mask = gt_mask[:, :, idx]
      if show_mask != 0:
        if objects[idx] == "skin":
          masked_image = apply_mask(masked_image, mask, skin_color)
        elif objects[idx] == "minor":
          masked_image = apply_mask(masked_image, mask, minor_defect_color)
        elif objects[idx] == "critical":
          masked_image = apply_mask(masked_image, mask, critical_defect_color)
      
      # Mask Polygon
      # Pad to ensure proper polygons for masks that touch image edges.
      padded_mask = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
      padded_mask[1:-1, 1:-1] = mask
      contours = find_contours(padded_mask, 0.5)
      for verts in contours:
        # Subtract the padding and flip (y, x) to (x, y)
        verts = np.fliplr(verts) - 1
        if objects[idx] == "skin":
          # Red
          p = Polygon(verts, facecolor="none", edgecolor=skin_color)
          ax.add_patch(p)
        elif objects[idx] == "minor":
          # Blue
          p = Polygon(verts, facecolor="none", edgecolor=minor_defect_color)
          ax.add_patch(p)
        elif objects[idx] == "critical":
          # Yellow
          p = Polygon(verts, facecolor="none", edgecolor=critical_defect_color)
          ax.add_patch(p)

    ax.imshow(masked_image.astype(np.uint8))
    plt.savefig(os.path.join(save_pred_dir, image_file_name),
                bbox_inches='tight', 
                pad_inches=-0.5,
                orientation= 'landscape')
    plt.show() 