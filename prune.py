#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
from random import randint
from gaussian_renderer import render, count_render
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from utils.proj_utils import project_xyz_to_pixels
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from utils.proj_utils import project_xyz_to_pixels
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.graphics_utils import getWorld2View2
from icecream import ic
import random
import copy
import gc
import numpy as np
from collections import defaultdict
import torchvision.utils as vutils
from PIL import Image

# from cuml.cluster import HDBSCAN


# def HDBSCAN_prune(gaussians, score_list, prune_percent):
#     # Ensure the tensor is on the GPU and detached from the graph
#     s, d = gaussians.get_xyz.shape
#     X_gpu = cp.asarray(gaussians.get_xyz.detach().cuda())

#     scores_gpu = cp.asarray(score_list.detach().cuda())
#     hdbscan = HDBSCAN(min_cluster_size = 100)
#     cluster_labels = hdbscan.fit_predict(X_gpu)
#     points_by_centroid = {}
#     ic("cluster_labels")
#     ic(cluster_labels.shape)
#     ic(cluster_labels)
#     for i, label in enumerate(cluster_labels):
#         if label not in points_by_centroid:
#             points_by_centroid[label] = []
#         points_by_centroid[label].append(i)
#     points_to_prune = []

#     for centroid_idx, point_indices in points_by_centroid.items():
#         # Skip noise points with label -1
#         if centroid_idx == -1:
#             continue
#         num_to_prune = int(cp.ceil(prune_percent * len(point_indices)))
#         if num_to_prune <= 3:
#             continue
#         point_indices_cp = cp.array(point_indices)
#         distances = scores_gpu[point_indices_cp].squeeze()
#         indices_to_prune = point_indices_cp[cp.argsort(distances)[:num_to_prune]]
#         points_to_prune.extend(indices_to_prune)
#     points_to_prune = np.array(points_to_prune)
#     mask = np.zeros(s, dtype=bool)
#     mask[points_to_prune] = True
#     # points_to_prune now contains the indices of the points to be pruned
#     return mask


# def uniform_prune(gaussians, k, score_list, prune_percent, sample = "k_mean"):
#     # get the farthest_point
#     D, I = None, None
#     s, d = gaussians.get_xyz.shape

#     if sample == "k_mean":
#         ic("k_mean")
#         n_iter = 200
#         verbose = False
#         kmeans = faiss.Kmeans(d, k=k, niter=n_iter, verbose=verbose, gpu=True)
#         kmeans.train(gaussians.get_xyz.detach().cpu().numpy())
#         # The cluster centroids can be accessed as follows
#         centroids = kmeans.centroids
#         D, I = kmeans.index.search(gaussians.get_xyz.detach().cpu().numpy(), 1)
#     else:
#         point_idx = farthest_point_sampler(torch.unsqueeze(gaussians.get_xyz, 0), k)
#         centroids = gaussians.get_xyz[point_idx,: ]
#         centroids = centroids.squeeze(0)
#         index = faiss.IndexFlatL2(d)
#         index.add(centroids.detach().cpu().numpy())
#         D, I = index.search(gaussians.get_xyz.detach().cpu().numpy(), 1)
#     points_to_prune = []
#     points_by_centroid = defaultdict(list)
#     for point_idx, centroid_idx in enumerate(I.flatten()):
#         points_by_centroid[centroid_idx.item()].append(point_idx)
#     for centroid_idx in points_by_centroid:
#         points_by_centroid[centroid_idx] = np.array(points_by_centroid[centroid_idx])
#     for centroid_idx, point_indices in points_by_centroid.items():
#         # Find the number of points to prune
#         num_to_prune = int(np.ceil(prune_percent * len(point_indices)))
#         if num_to_prune <= 3:
#             continue
#         distances = score_list[point_indices].squeeze().cpu().detach().numpy()
#         indices_to_prune = point_indices[np.argsort(distances)[:num_to_prune]]
#         points_to_prune.extend(indices_to_prune)
#     # Convert the list to an array
#     points_to_prune = np.array(points_to_prune)
#     mask = np.zeros(s, dtype=bool)
#     mask[points_to_prune] = True
#     return mask



def calculate_v_imp_score(gaussians, imp_list, v_pow):
    """
    :param gaussians: A data structure containing Gaussian components with a get_scaling method.
    :param imp_list: The importance scores for each Gaussian component.
    :param v_pow: The power to which the volume ratios are raised.
    :return: A list of adjusted values (v_list) used for pruning.
    """
    # Calculate the volume of each Gaussian component
    volume = torch.prod(gaussians.get_scaling, dim=1)
    # Determine the kth_percent_largest value
    index = int(len(volume) * 0.9)
    sorted_volume, _ = torch.sort(volume, descending=True)
    kth_percent_largest = sorted_volume[index]
    # Calculate v_list
    v_list = torch.pow(volume / kth_percent_largest, v_pow)
    v_list = v_list * imp_list
    return v_list




def prune_list(gaussians, scene, pipe, background):
    viewpoint_stack = scene.getTrainCameras().copy()
    gaussian_list, imp_list = None, None
    viewpoint_cam = viewpoint_stack.pop()
    render_pkg = count_render(viewpoint_cam, gaussians, pipe, background)
    gaussian_list, imp_list = (
        render_pkg["gaussians_count"],
        render_pkg["important_score"],
    )

    # ic(dataset.model_path)
    for iteration in range(len(viewpoint_stack)):
        # Pick a random Camera
        # prunning
        viewpoint_cam = viewpoint_stack.pop()
        render_pkg = count_render(viewpoint_cam, gaussians, pipe, background)
        # image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        gaussians_count, important_score = (
            render_pkg["gaussians_count"].detach(),
            render_pkg["important_score"].detach(),
        )
        gaussian_list += gaussians_count
        imp_list += important_score
        gc.collect()
    return gaussian_list, imp_list

def prune_edge_base(gaussians, scene, dataset, edge_dmaps, save_path):
    print(f"\n[Global Pruning] Starting global scan over {len(scene.getTrainCameras())} views...")
    
    n_points = gaussians.get_xyz.shape[0]
    global_importance_scores = torch.full((n_points,), 0.0, device="cuda")
    viewpoint_stack = scene.getTrainCameras().copy()

    for iteration in tqdm(range(len(viewpoint_stack))):
        # Pick a random Camera
        viewpoint_cam = viewpoint_stack.pop()
        
        # project points to pixels
        uv, inb = project_xyz_to_pixels(gaussians.get_xyz, viewpoint_cam)

        # load edge dmap for the current viewpoint
        edge_map = edge_dmaps[id(viewpoint_cam)]

        # found point in visible area
        visible_uv = uv[inb].cpu().numpy().astype(int) # (k, 2)

        # extract edge distance map values
        current_dist = edge_map[visible_uv[:, 1], visible_uv[:, 0]]  # (k, )

        # global_importance_scores[inb] = torch.min(global_importance_scores[inb], current_dist)
        global_importance_scores[inb] += current_dist

        ############## For projection debug ############## --> Before
        debug_dir = os.path.join(dataset.model_path, save_path)
        os.makedirs(debug_dir, exist_ok=True)

        iteration_str = str(iteration).zfill(7)
        base_name = os.path.splitext(os.path.basename(viewpoint_cam.image_name))[0]
        save_path_proj = os.path.join(debug_dir, f"{iteration_str}_{base_name}_projection.png")
        save_path_orig = os.path.join(debug_dir, f"{iteration_str}_{base_name}_original.png")

        H = viewpoint_cam.image_height
        W = viewpoint_cam.image_width

        # 1. 검은색 빈 캔버스(이미지)를 만듭니다. (H, W)
        projection_image = np.zeros((H, W), dtype=np.uint8)

        # 2. 'inb' (in-bounds) 마스크가 True인 가우시안만 필터링합니다.
        #    이것이 "프러스텀 컬링"을 검증합니다.
        visible_uv = uv[inb].cpu().numpy().astype(int) # (k, 2)

        # 3. 캔버스의 해당 픽셀을 흰색(255)으로 칠합니다.
        projection_image[visible_uv[:, 1], visible_uv[:, 0]] = 255

        # 4. 투영된 포인트 클라우드 이미지를 저장합니다.
        img_pil = Image.fromarray(projection_image, 'L')
        img_pil.save(save_path_proj)

        # 6. 비교를 위해 원본 이미지도 바로 옆에 저장합니다.
        vutils.save_image(viewpoint_cam.original_image, save_path_orig)

        # print(f"[DEBUG] Saved projection test image to: {save_path_proj}")
        ##################################################
    
    print(f"[Debug] Min dist stats: Min={global_importance_scores.min():.4f}, Mean={global_importance_scores.mean():.4f}, Max={global_importance_scores.max():.4f}")

    # global_importance scroes 분포 시각화
    debug_dir = os.path.join(dataset.model_path)
    os.makedirs(debug_dir, exist_ok=True)
    save_path_hist = os.path.join(debug_dir, f"global_importance_histogram.png")
    import matplotlib.pyplot as plt
    plt.figure()
    plt.hist(global_importance_scores.cpu().numpy(), bins=100, color='blue', alpha=0.7)
    plt.title('Global Importance Scores Distribution')
    plt.xlabel('Importance Score')
    plt.ylabel('Frequency')
    plt.savefig(save_path_hist)
    plt.close()

    return global_importance_scores

