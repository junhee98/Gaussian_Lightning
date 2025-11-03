# utils/edge_utils.py
import cv2, torch, numpy as np

def build_edge_distance_map(rgb_tensor_chw_01):
    # rgb: (3,H,W) [0,1] torch
    img = (rgb_tensor_chw_01.permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    inv = (edges == 0).astype(np.uint8)
    dist = cv2.distanceTransform(inv, cv2.DIST_L2, 3).astype(np.float32)
    if dist.max() > 0:
        dist /= (dist.max() + 1e-6)  # [0,1]
    return torch.from_numpy(dist)  # (H,W) torch
