import torch
import numpy as np
from tqdm import tqdm

from .distance_function import IoU


"""
SOURCE: https://medium.com/@yerdaulet.zhumabay/generating-anchor-boxes-by-k-means-82f11c690b82
"""
def KMeans(bboxes:torch.tensor, k:int, stop_iter=5):
    pbar = tqdm(total=stop_iter)
    rows = bboxes.shape[0]
    distances = torch.empty((rows, k))
    last_clusters = torch.zeros((rows, ))

    cluster_indxs = np.random.choice(rows, k, replace=False) # choose unique indexs in rows
    clusters = bboxes[cluster_indxs].clone()

    iteration = 0
    while True:
        # calculate the distances 
        distances = IoU(bboxes, clusters)

        nearest_clusters = torch.argmax(distances, dim=1) # 0, 1, 2 ... K   

        if (last_clusters == nearest_clusters).all(): # break if nothing changes
            iteration += 1
            pbar.update(1)
            if iteration == stop_iter:
                break
        else:
            iteration = 0
        # Take the mean and step for cluster coordiantes 
        for cluster in range(k):
            clusters[cluster] = torch.mean(bboxes[nearest_clusters == cluster], axis=0)

        last_clusters = nearest_clusters.clone()
    pbar.close()
    return clusters, distances