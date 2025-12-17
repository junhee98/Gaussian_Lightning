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
from utils.edge_utils import build_texture_based_distance_map, build_edge_distance_map
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
        gaussians.training_setup(opt)
        gaussians.max_radii2D = torch.zeros((gaussians.get_xyz.shape[0]), device="cuda")
        
    else:
        raise ValueError("A checkpoint file or a pointcloud is required to proceed.")


    # build edge distance maps for all training views
    edge_dmaps = {}  
    edge_score = {}
    edge_score_views = None
    densify_count = 0
    for cam in scene.getTrainCameras():
        d = build_texture_based_distance_map(cam.original_image)
        edge_score[id(cam)] = d.sum().item()
        # find high edge_score and low edge_score views
        sorted_scores = sorted(edge_score.items(), key=lambda x: x[1])
        # select median 5 views
        median_index = len(sorted_scores) // 2
        edge_score_views = sorted_scores[median_index - 2 : median_index + 3]
        edge_dmaps[id(cam)] = d.cuda(non_blocking=True)
    
    ################## For Edge map debug ##################
    # edge_map_debug_dir = os.path.join(dataset.model_path, "edge_map_debug")
    # os.makedirs(edge_map_debug_dir, exist_ok=True)
    # print(f"[DEBUG] Saving edge map visualizations to: {edge_map_debug_dir}")

    # edge_dmaps = {}
    # for cam in scene.getTrainCameras():
    #     d = build_texture_based_distance_map(cam.original_image)
    #     edge_dmaps[id(cam)] = d.cuda(non_blocking=True)

    #     base_name = os.path.splitext(os.path.basename(cam.image_name))[0]

    #     # Distance map visualization
    #     d_np = d.detach().cpu().numpy()
    #     d_img_array = (d_np * 255.0).astype(np.uint8)
    #     img_d = Image.fromarray(d_img_array, 'L')
    #     save_path_d = os.path.join(edge_map_debug_dir, f"distmap_{base_name}.png")
    #     img_d.save(save_path_d)
    #########################################################
    
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    gaussians.scheduler = ExponentialLR(gaussians.optimizer, gamma=0.95)
    lr_stack = []

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
        
        lr = gaussians.optimizer.param_groups[0]["lr"]
        lr_stack.append((iteration, lr))

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

        # # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

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
                print("Before prune iteration, number of gaussians: " + str(len(gaussians.get_xyz)))
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
                else:
                    raise Exception("Unsupportive pruning method")

                print("After prune iteration, number of gaussians: " + str(len(gaussians.get_xyz)))

            DENSIFY_PERIOD = 200
            if iteration <= args.densify_iteration[-1] and densify_count >= DENSIFY_PERIOD:
                if edge_score[id(viewpoint_cam)] in [score for (cam_id, score) in edge_score_views]:
                    print(f"\n[ITER {iteration}] Densification triggered on view with edge score {edge_score[id(viewpoint_cam)]:.4f}")
                    densify_count = 0
                    # delete score to avoid repeated densification on same view
                    del edge_score_views[[item[0] for item in edge_score_views].index(id(viewpoint_cam))]

                    ############ For Debugging (Which view selected?) #######################
                    # debug_dir = os.path.join(dataset.model_path, "densify_trigger_debug")
                    # os.makedirs(debug_dir, exist_ok=True)

                    # iteration_str = str(iteration).zfill(7)
                    # base_name = os.path.splitext(os.path.basename(viewpoint_cam.image_name))[0]
                    # save_path = os.path.join(debug_dir, f"{iteration_str}_{base_name}.png")

                    # vutils.save_image(viewpoint_cam.original_image, save_path)
                    # print(f"\n[DEBUG] Saved densify trigger image to: {save_path}")
                    ##################################################

                    gaussians.max_radii2D[visibility_filter] = torch.max(
                        gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
                    )
                    gaussians.add_densification_stats(
                        viewspace_point_tensor, visibility_filter
                    )
                    
                    uv, inb = project_xyz_to_pixels(gaussians.get_xyz, viewpoint_cam)

                    ############## For projection debug ############## --> Before
                    # debug_dir = os.path.join(dataset.model_path, "projection_test_debug")
                    # os.makedirs(debug_dir, exist_ok=True)

                    # iteration_str = str(iteration).zfill(7)
                    # base_name = os.path.splitext(os.path.basename(viewpoint_cam.image_name))[0]
                    # save_path_proj = os.path.join(debug_dir, f"{iteration_str}_{base_name}_projection.png")
                    # save_path_orig = os.path.join(debug_dir, f"{iteration_str}_{base_name}_original.png")

                    # H = viewpoint_cam.image_height
                    # W = viewpoint_cam.image_width

                    # projection_image = np.zeros((H, W), dtype=np.uint8)

                    # visible_uv = uv[inb].cpu().numpy().astype(int) # (k, 2)

                    # projection_image[visible_uv[:, 1], visible_uv[:, 0]] = 255

                    # img_pil = Image.fromarray(projection_image, 'L')
                    # img_pil.save(save_path_proj)

                    # vutils.save_image(viewpoint_cam.original_image, save_path_orig)

                    # print(f"[DEBUG] Saved projection test image to: {save_path_proj}")
                    ##################################################


                    dmap = edge_dmaps[id(viewpoint_cam)]
                    d = torch.full((gaussians.get_xyz.shape[0],), 1e6, device='cuda')
                    ui = uv[inb,0].long()
                    vi = uv[inb,1].long()
                    d[inb] = dmap[vi, ui]
                    edge_mask = 1.0 - d

                    ############## For edge mask debug ##############
                    # debug_dir = os.path.join(dataset.model_path, "projected_edge_mask_debug")
                    # os.makedirs(debug_dir, exist_ok=True)
                    
                    # iteration_str = str(iteration).zfill(7)
                    # base_name = os.path.splitext(os.path.basename(viewpoint_cam.image_name))[0]
                    # save_path_heatmap = os.path.join(debug_dir, f"{iteration_str}_{base_name}_projected_heatmap.png")

                    # H = viewpoint_cam.image_height
                    # W = viewpoint_cam.image_width

                    # projected_mask_2d = torch.zeros((H, W), dtype=torch.float32, device='cpu')

                    # valid_uv = uv[inb].cpu()
                    # valid_edge_mask = edge_mask[inb].cpu() # (k,)

                    # valid_y = valid_uv[:, 1].long()
                    # valid_x = valid_uv[:, 0].long()

                    # projected_mask_2d[valid_y, valid_x] = valid_edge_mask

                    # plt.figure(figsize=(W/100, H/100), dpi=100)
                    # plt.imshow(projected_mask_2d.numpy(), cmap='hot', vmin=0, vmax=1)
                    # plt.axis('off')
                    # plt.tight_layout()
                    # plt.savefig(save_path_heatmap, bbox_inches='tight', pad_inches=0)
                    # plt.close()
                    ##################################################


                    size_threshold = (
                        20 if iteration > opt.opacity_reset_interval else None
                    )


                    gaussians.densify_and_prune_edge_aware(
                        opt.densify_grad_threshold,  
                        0.005,             
                        scene.cameras_extent,        
                        size_threshold,         
                        edge_mask,                   
                        max_opacity_prune_non_edge=0.5 
                    )

                    

                    ############## For projection debug ##############
                    # uv, inb = project_xyz_to_pixels(gaussians.get_xyz, viewpoint_cam)
                    # debug_dir = os.path.join(dataset.model_path, "projection_test_debug_after_densify")
                    # os.makedirs(debug_dir, exist_ok=True)

                    # iteration_str = str(iteration).zfill(7)
                    # base_name = os.path.splitext(os.path.basename(viewpoint_cam.image_name))[0]
                    # save_path_proj = os.path.join(debug_dir, f"{iteration_str}_{base_name}_projection.png")
                    # save_path_orig = os.path.join(debug_dir, f"{iteration_str}_{base_name}_original.png")

                    # H = viewpoint_cam.image_height
                    # W = viewpoint_cam.image_width

                    # projection_image = np.zeros((H, W), dtype=np.uint8)

                    # visible_uv = uv[inb].cpu().numpy().astype(int) # (k, 2)

                    # projection_image[visible_uv[:, 1], visible_uv[:, 0]] = 255

                    # img_pil = Image.fromarray(projection_image, 'L')
                    # img_pil.save(save_path_proj)

                    # vutils.save_image(viewpoint_cam.original_image, save_path_orig)

                    # print(f"[DEBUG] Saved projection test image to: {save_path_proj}")
                    ##################################################


                    ic("after")
                    ic(gaussians.get_xyz.shape)
                    ic(len(gaussians.optimizer.param_groups[0]['params'][0]))
                    ic(gaussians.max_sh_degree)

            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            densify_count += 1
    
    plot_path = os.path.join(dataset.model_path, "learning_rate_schedule_v5.png")
    iterations, lrs = zip(*lr_stack)
    plt.figure(figsize=(12, 6))
    plt.plot(iterations, lrs)
    plt.title(f"XYZ Learning Rate Schedule (Final LR: {lrs[-1]:.2e})")
    plt.xlabel("Iteration")
    plt.ylabel("Learning Rate (Log Scale)")
    plt.yscale('log') # LR은 로그 스케일로 보는 것이 좋습니다.
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    print(f"[INFO] LR graph saved to {plot_path}")


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
        "--test_iterations", nargs="+", type=int, default=[30_001, 30_002, 35_000] ## 35_000
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
    parser.add_argument("--densify_iteration", nargs="+", type=int, default=[30_500, 33_000]) # 30_500, 33_000
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
