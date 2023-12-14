import numpy as np
import skimage.draw


def extract_anno_gt(image_path, gt_polygons, gt_objects, classes_map):
    gt_image = skimage.io.imread(image_path)

    print(f"[GROUND TRUTH IMG SHAPE]: {gt_image.shape}")

    gt_label_ids = [classes_map[a] for a in gt_objects]
    gt_mask = np.zeros([480, 640, len(gt_polygons)], 
                       dtype=np.uint8)

    for i, p in enumerate(gt_polygons):
        # Get indexes of pixels inside the polygon and set them to 1
        rr, cc = skimage.draw.polygon(p["all_points_y"], p["all_points_x"])
        rr[rr > gt_mask.shape[0]-1] = gt_mask.shape[0] - 1
        cc[cc > gt_mask.shape[1]-1] = gt_mask.shape[1] - 1           
        gt_mask[rr, cc, i] = 1
    return gt_image, gt_mask, np.asarray(gt_label_ids)