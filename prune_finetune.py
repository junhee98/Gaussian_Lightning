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
import re
import torch
from random import randint
from utils.proj_utils import project_xyz_to_pixels
from utils.loss_utils import l1_loss, ssim
from lpipsPyTorch import lpips
from gaussian_renderer import render, network_gui, count_render
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from utils.edge_utils import build_edge_distance_map
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import numpy as np
import torchvision.utils as vutils
from matplotlib import pyplot as plt

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
from icecream import ic
import random
import copy
import gc
from os import makedirs
from prune import prune_list, calculate_v_imp_score
import torchvision
from torch.optim.lr_scheduler import ExponentialLR
import csv
from utils.logger_utils import training_report, prepare_output_and_logger
from PIL import Image

to_tensor = (
    lambda x: x.to("cuda")
    if isinstance(x, torch.Tensor)
    else torch.Tensor(x).to("cuda")
)
img2mse = lambda x, y: torch.mean((x - y) ** 2)
mse2psnr = lambda x: -10.0 * torch.log(x) / torch.log(to_tensor([10.0]))


def valid_render_pkg(pkg):
    if not isinstance(pkg, dict):
        return False, "not a dict"

    required = ["render", "viewspace_points", "visibility_filter", "radii"]
    missing = [k for k in required if k not in pkg]
    if missing:
        return False, f"missing keys: {missing}"

    # 값이 None이거나 타입/디바이스가 이상한지 체크
    if pkg["render"] is None:
        return False, "render is None"
    if pkg["viewspace_points"] is None or pkg["radii"] is None:
        return False, "viewspace_points or radii is None"
    if pkg["visibility_filter"] is None:
        return False, "visibility_filter is None"

    # 선택: 텐서 타입/디바이스/shape 간단 검증
    try:
        vsp = pkg["viewspace_points"]
        vf  = pkg["visibility_filter"]
        r   = pkg["radii"]
        # visibility_filter와 radii 길이 일치 여부
        if hasattr(vf, "shape") and hasattr(r, "shape") and vf.shape != r.shape:
            return False, f"shape mismatch vf={vf.shape}, radii={r.shape}"
    except Exception as e:
        return False, f"type/device check failed: {e}"

    return True, "ok"


def training(
    dataset,
    opt,
    pipe,
    testing_iterations,
    saving_iterations,
    checkpoint_iterations,
    checkpoint,
    debug_from,
    args,
):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    if checkpoint:
        gaussians.training_setup(opt)
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
    elif args.start_pointcloud:
        gaussians.load_ply(args.start_pointcloud)
        ic(gaussians.get_xyz.shape)
        # ic(gaussians.optimizer.param_groups["xyz"].shape)
        gaussians.training_setup(opt)
        gaussians.max_radii2D = torch.zeros((gaussians.get_xyz.shape[0]), device="cuda")
        
    else:
        raise ValueError("A checkpoint file or a pointcloud is required to proceed.")

    # build edge distance maps for all training views
    edge_dmaps = {}  # {cam_id: torch(H,W) on CUDA}
    for cam in scene.getTrainCameras():
        d = build_edge_distance_map(cam.original_image)  # (H,W) CPU
        edge_dmaps[id(cam)] = d.cuda(non_blocking=True)

    # ################## For Edge map debug ##################
    # edge_map_debug_dir = os.path.join(dataset.model_path, "edge_map_debug")
    # os.makedirs(edge_map_debug_dir, exist_ok=True)
    # print(f"[DEBUG] Saving edge map visualizations to: {edge_map_debug_dir}")

    # edge_dmaps = {}
    # for cam in scene.getTrainCameras():
    #     d = build_edge_distance_map(cam.original_image)  # (H,W) CPU
    #     edge_dmaps[id(cam)] = d.cuda(non_blocking=True)

    #     base_name = os.path.splitext(os.path.basename(cam.image_name))[0]

    #     # 거리 맵 (Distance Map) 시각화 (0=검정색, 멀수록=흰색)
    #     d_np = d.numpy()
    #     print(f"Min: {d_np.min()}, Max: {d_np.max()}")
    #     # [0, max_dist] 범위를 [0, 255] 범위로 정규화
    #     d_img_array = (d_np * 255.0).astype(np.uint8)
    #     img_d = Image.fromarray(d_img_array, 'L') # 'L' = 흑백
    #     save_path_d = os.path.join(edge_map_debug_dir, f"distmap_{base_name}.png")
    #     img_d.save(save_path_d)
    # #########################################################
    
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    gaussians.scheduler = ExponentialLR(gaussians.optimizer, gamma=0.95)

    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                (
                    custom_cam,
                    do_training,
                    pipe.convert_SHs_python,
                    pipe.compute_cov3D_python,
                    keep_alive,
                    scaling_modifer,
                ) = network_gui.receive()
                if custom_cam != None:
                    net_image = render(
                        custom_cam, gaussians, pipe, background, scaling_modifer
                    )["render"]
                    net_image_bytes = memoryview(
                        (torch.clamp(net_image, min=0, max=1.0) * 255)
                        .byte()
                        .permute(1, 2, 0)
                        .contiguous()
                        .cpu()
                        .numpy()
                    )
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and (
                    (iteration < int(opt.iterations)) or not keep_alive
                ):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()
        if iteration % 400 == 0:
            gaussians.scheduler.step()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii = (
            render_pkg["render"],
            render_pkg["viewspace_points"],
            render_pkg["visibility_filter"],
            render_pkg["radii"],
        )

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (
            1.0 - ssim(image, gt_image)
        )

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 1000 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(1000)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save

            if iteration in saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            if iteration in checkpoint_iterations:
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                if not os.path.exists(scene.model_path):
                    os.makedirs(scene.model_path)
                torch.save(
                    (gaussians.capture(), iteration),
                    scene.model_path + "/chkpnt" + str(iteration) + ".pth",
                )
                
                if iteration == checkpoint_iterations[-1]:
                    gaussian_list, imp_list = prune_list(gaussians, scene, pipe, background)
                    v_list = calculate_v_imp_score(gaussians, imp_list, args.v_pow)
                    np.savez(os.path.join(scene.model_path,"imp_score"), v_list.cpu().detach().numpy()) 


            training_report(
                tb_writer,
                iteration,
                Ll1,
                loss,
                l1_loss,
                iter_start.elapsed_time(iter_end),
                testing_iterations,
                scene,
                render,
                (pipe, background),
            )

            if iteration in args.prune_iterations:
                ic("Before prune iteration, number of gaussians: " + str(len(gaussians.get_xyz)))
                i = args.prune_iterations.index(iteration)
                gaussian_list, imp_list = prune_list(gaussians, scene, pipe, background)

                if args.prune_type == "important_score":
                    gaussians.prune_gaussians(
                        (args.prune_decay**i) * args.prune_percent, imp_list
                    )
                elif args.prune_type == "v_important_score":
                    # normalize scale
                    v_list = calculate_v_imp_score(gaussians, imp_list, args.v_pow)
                    gaussians.prune_gaussians(
                        (args.prune_decay**i) * args.prune_percent, v_list
                    )
                elif args.prune_type == "max_v_important_score":
                    v_list = imp_list * torch.max(gaussians.get_scaling, dim=1)[0]
                    gaussians.prune_gaussians(
                        (args.prune_decay**i) * args.prune_percent, v_list
                    )
                elif args.prune_type == "count":
                    gaussians.prune_gaussians(
                        (args.prune_decay**i) * args.prune_percent, gaussian_list
                    )
                elif args.prune_type == "opacity":
                    gaussians.prune_gaussians(
                        (args.prune_decay**i) * args.prune_percent,
                        gaussians.get_opacity.detach(),
                    )
                # TODO(release different pruning method) --> This is from LightGaussian
                # elif args.prune_type == "HDBSCAN":
                #     masks = HDBSCAN_prune(gaussians, imp_list, (args.prune_decay**i)*args.prune_percent)
                #     gaussians.prune_points(masks)
                # # elif args.prune_type == "v_important_score":
                # #     imp_list *
                # elif args.prune_type == "two_step":
                #     if i == 0:
                #         volume = torch.prod(gaussians.get_scaling, dim = 1)
                #         index = int(len(volume) * 0.9)
                #         sorted_volume, sorted_indices = torch.sort(volume, descending=True, dim=0)
                #         kth_percent_largest = sorted_volume[index]
                #         v_list = torch.pow(volume/kth_percent_largest, args.v_pow)
                #         v_list = v_list * imp_list
                #         gaussians.prune_gaussians((args.prune_decay**i)*args.prune_percent, v_list)
                #     else:
                #         k = 5^(1*i) * 100
                #         masks = uniform_prune(gaussians, k, imp_list, 0.3, "k_mean")
                #         gaussians.prune_points(masks)
                # else:
                #     k = len(gaussians.get_xyz)//500 * i
                #     masks = uniform_prune(gaussians, k, imp_list, (args.prune_decay**i)*args.prune_percent, args.prune_type)
                #     gaussians.prune_points(masks)
                # gaussians.prune_gaussians(args.prune_percent, imp_list)
                # gaussians.optimizer.zero_grad(set_to_none = True) #hachy way to maintain grad
                # if (iteration in args.opacity_prune_iterations):
                #         gaussians.prune_opacity(0.05)
                else:
                    raise Exception("Unsupportive pruning method")

                ic("After prune iteration, number of gaussians: " + str(len(gaussians.get_xyz)))

            DENSIFY_PERIOD = 500
            if iteration < args.densify_iteration[-1] and iteration >= args.densify_iteration[0] and iteration % DENSIFY_PERIOD == 0:
                
                # ############ For Debugging (Which view selected?) #######################
                # debug_dir = os.path.join(dataset.model_path, "densify_trigger_debug")
                # os.makedirs(debug_dir, exist_ok=True)

                # iteration_str = str(iteration).zfill(7)
                # base_name = os.path.splitext(os.path.basename(viewpoint_cam.image_name))[0]
                # save_path = os.path.join(debug_dir, f"{iteration_str}_{base_name}.png")

                # vutils.save_image(viewpoint_cam.original_image, save_path)
                # print(f"\n[DEBUG] Saved densify trigger image to: {save_path}")
                # ##################################################

                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
                )
                gaussians.add_densification_stats(
                    viewspace_point_tensor, visibility_filter
                )
                
                # 현재 뽑은 viewpoint_cam 기준 엣지 마스크 산출
                uv, inb = project_xyz_to_pixels(gaussians.get_xyz, viewpoint_cam)

                # ############## For projection debug ##############
                # debug_dir = os.path.join(dataset.model_path, "projection_test_debug")
                # os.makedirs(debug_dir, exist_ok=True)

                # iteration_str = str(iteration).zfill(7)
                # base_name = os.path.splitext(os.path.basename(viewpoint_cam.image_name))[0]
                # save_path_proj = os.path.join(debug_dir, f"{iteration_str}_{base_name}_projection.png")
                # save_path_orig = os.path.join(debug_dir, f"{iteration_str}_{base_name}_original.png")

                # H = viewpoint_cam.image_height
                # W = viewpoint_cam.image_width

                # # 1. 검은색 빈 캔버스(이미지)를 만듭니다. (H, W)
                # projection_image = np.zeros((H, W), dtype=np.uint8)

                # # 2. 'inb' (in-bounds) 마스크가 True인 가우시안만 필터링합니다.
                # #    이것이 "프러스텀 컬링"을 검증합니다.
                # visible_uv = uv[inb].cpu().numpy().astype(int) # (k, 2)

                # # 3. 캔버스의 해당 픽셀을 흰색(255)으로 칠합니다.
                # projection_image[visible_uv[:, 1], visible_uv[:, 0]] = 255

                # # 4. 투영된 포인트 클라우드 이미지를 저장합니다.
                # img_pil = Image.fromarray(projection_image, 'L')
                # img_pil.save(save_path_proj)

                # # 6. 비교를 위해 원본 이미지도 바로 옆에 저장합니다.
                # vutils.save_image(viewpoint_cam.original_image, save_path_orig)

                # print(f"[DEBUG] Saved projection test image to: {save_path_proj}")
                # ##################################################


                dmap = edge_dmaps[id(viewpoint_cam)]
                d = torch.full((gaussians.get_xyz.shape[0],), 1e6, device='cuda')
                ui = uv[inb,0].long()
                vi = uv[inb,1].long()
                d[inb] = dmap[vi, ui]
                # 엣지 가까울수록 1에 가깝게: exp(-d/tau)
                edge_mask = torch.exp(-d/0.1)   # tau=0.1 예시 (장면에 맞춰 튜닝)

                # ############## For edge mask debug ##############
                # debug_dir = os.path.join(dataset.model_path, "projected_edge_mask_debug")
                # os.makedirs(debug_dir, exist_ok=True)
                
                # iteration_str = str(iteration).zfill(7)
                # base_name = os.path.splitext(os.path.basename(viewpoint_cam.image_name))[0]
                # save_path_heatmap = os.path.join(debug_dir, f"{iteration_str}_{base_name}_projected_heatmap.png")

                # H = viewpoint_cam.image_height
                # W = viewpoint_cam.image_width

                # # (a) 2D 캔버스 생성
                # projected_mask_2d = torch.zeros((H, W), dtype=torch.float32, device='cpu')

                # # (b) 뷰 안에 보이는(inb) 가우시안들의 2D 좌표와 엣지 가중치 추출
                # valid_uv = uv[inb].cpu()
                # valid_edge_mask = edge_mask[inb].cpu() # (k,)

                # valid_y = valid_uv[:, 1].long()
                # valid_x = valid_uv[:, 0].long()

                # # (c) 캔버스에 엣지 가중치 값을 "뿌리기"
                # projected_mask_2d[valid_y, valid_x] = valid_edge_mask

                # # (d) 히트맵으로 저장 ('hot' 컬러맵: 0=검정, 1=밝은 흰색)
                # plt.figure(figsize=(W/100, H/100), dpi=100)
                # plt.imshow(projected_mask_2d.numpy(), cmap='hot', vmin=0, vmax=1)
                # plt.axis('off')
                # plt.tight_layout()
                # plt.savefig(save_path_heatmap, bbox_inches='tight', pad_inches=0)
                # plt.close()
                # ##################################################


                size_threshold = (
                    20 if iteration > opt.opacity_reset_interval else None
                )


                gaussians.densify_and_prune_edge_aware(
                    opt.densify_grad_threshold,  # max_grad
                    0.005,             # min_opacity (옵션 파라미터 이름 확인 필요)
                    scene.cameras_extent,        # extent
                    size_threshold,         # max_screen_size (옵션 파라미터 이름 확인 필요)
                    edge_mask,                   # edge_mask_float
                    max_opacity_prune_non_edge=0.5     # 엣지 prune 임계값 (튜닝 가능) # 0.1
                )

                # gaussians.densify(opt.densify_grad_threshold, scene.cameras_extent)
            
                ic("after")
                ic(gaussians.get_xyz.shape)
                ic(len(gaussians.optimizer.param_groups[0]['params'][0]))

            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6009)
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument(
        "--test_iterations", nargs="+", type=int, default=[30_001, 30_002, 35_000]
    )
    parser.add_argument(
        "--save_iterations", nargs="+", type=int, default=[35_000]
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--checkpoint_iterations", nargs="+", type=int, default=[35_000]
    )

    parser.add_argument("--prune_iterations", nargs="+", type=int, default=[30_001])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--start_pointcloud", type=str, default=None)
    parser.add_argument("--prune_percent", type=float, default=0.1)
    parser.add_argument("--prune_decay", type=float, default=1)
    parser.add_argument(
        "--prune_type", type=str, default="important_score"
    )  # k_mean, farther_point_sample, important_score
    parser.add_argument("--v_pow", type=float, default=0.1)
    parser.add_argument("--densify_iteration", nargs="+", type=int, default=[30_500, 33_000]) # 33_500
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(
        lp.extract(args),
        op.extract(args),
        pp.extract(args),
        args.test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint,
        args.debug_from,
        args,
    )

    # All done
    print("\nTraining complete.")
