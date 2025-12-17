# utils/edge_utils.py
import cv2, torch, numpy as np
import torch.nn.functional as F
import torchvision.transforms.functional as TF

def build_edge_distance_map(rgb_tensor_chw_01):
    img = (rgb_tensor_chw_01.permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 10, 200)
    inv = (edges == 0).astype(np.uint8)
    dist = cv2.distanceTransform(inv, cv2.DIST_L2, 3).astype(np.float32)
    if dist.max() > 0:
        dist /= (dist.max() + 1e-6)
    return torch.from_numpy(dist)


def build_texture_based_distance_map(rgb_tensor_chw_01, kernel_size=3, texture_threshold=0.025):
    
    x = rgb_tensor_chw_01.unsqueeze(0) 
    gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]

    padding = kernel_size // 2
    mean = F.avg_pool2d(gray, kernel_size=kernel_size, stride=1, padding=padding)
    mean_sq = F.avg_pool2d(gray**2, kernel_size=kernel_size, stride=1, padding=padding)
    variance = torch.clamp(mean_sq - mean**2, min=0)
    std = torch.sqrt(variance).squeeze()

    std_np = std.detach().cpu().numpy()
    
    binary_mask = np.ones_like(std_np, dtype=np.uint8)
    binary_mask[std_np > texture_threshold] = 0

    dist = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 3).astype(np.float32)

    if dist.max() > 0:
        dist /= (dist.max() + 1e-6)

    return torch.from_numpy(dist).to(rgb_tensor_chw_01.device)