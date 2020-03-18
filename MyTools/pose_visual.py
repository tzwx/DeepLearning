# ------------------------------------------------------------------------------
# # @Time    : 2020/3/18 下午 8:05
# # @Author  : fry
# @FileName: pose_visual.py
# ------------------------------------------------------------------------------

import cv2
import numpy as np
from .heatMap_to_coordinates import get_final_preds
from .heatMap_to_coordinates import _box2cs



# Method 1
def pose_visual_circle(image, heatmap, bbox):
    image = cv2.imread(image)
    c, s = _box2cs(bbox, 720, 1280) # bbox, image_height, image_width
    # compute coordinate
    preds, maxvals = get_final_preds(
        heatmap.clone().cpu().numpy(), np.asarray([c]), np.asarray([s]))
    image = image.copy()
    for mat in preds[0]:
        x, y = int(mat[0]), int(mat[1])
        cv2.circle(image, (x, y), 2, (255, 0, 0), 2)




