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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
from utils.edge_utils import build_edge_distance_map, soft_edge_map


def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()


def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()


def gaussian(window_size, sigma):
    gauss = torch.Tensor(
        [
            exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
            for x in range(window_size)
        ]
    )
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(
        _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    )
    return window


def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    )
    sigma2_sq = (
        F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    )
    sigma12 = (
        F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel)
        - mu1_mu2
    )

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def img2mse(x, y, mask=None):
    if mask is None:
        return torch.mean((x - y) ** 2)
    else:
        return torch.sum((x * mask - y * mask) ** 2) / (torch.sum(mask) + 1e-5)


def img2mae(x, y, mask=None):
    if mask is None:
        return torch.mean(torch.abs(x - y))
    else:
        return torch.sum(torch.abs(x * mask - y * mask)) / (torch.sum(mask) + 1e-5)
    

def edge_distance_loss(pred_bchw_01, gt_bchw_01, d_trunc=None, weight=1.0, eps=1e-6):
    B, C, H, W = pred_bchw_01.shape
    # 1) GT 거리맵(각 배치별 계산)
    Dg_list = []
    for b in range(B):
        Dg = build_edge_distance_map(gt_bchw_01[b].detach())  # (H,W), no grad
        if d_trunc is not None:
            Dg = torch.clamp(Dg, 0.0, d_trunc) / (d_trunc + 1e-6)
        Dg_list.append(Dg)
    D_gt = torch.stack(Dg_list, dim=0).to(pred_bchw_01.device).unsqueeze(1)  # [B,1,H,W], [0,1]

    # 2) 예측 소프트 엣지 확률 (grad 흐름 O)
    E_pred = soft_edge_map(pred_bchw_01)                                     # [B,1,H,W]

    # 3) 정규화된 단방향 손실
    num = (E_pred * D_gt).sum(dim=(1,2,3))
    den = (E_pred.sum(dim=(1,2,3)) + eps)
    L = (num / den).mean()
    
def tv_loss(image):
    """
    Total Variation Loss.
    Calculates the sum of squared differences between adjacent pixels.
    Handles images of shape [C, H, W] or [B, C, H, W].
    """
    # ... (Ellipsis)는 [C, H, W]와 [B, C, H, W] 모두를 처리합니다.
    # 이 로직은 문제가 없었습니다.
    w_variance = torch.sum(torch.pow(image[..., :, 1:] - image[..., :, :-1], 2))
    h_variance = torch.sum(torch.pow(image[..., 1:, :] - image[..., :-1, :], 2))

    # --- [수정된 부분] ---
    # 분모를 텐서의 전체 요소 수로 나눕니다.
    # 이게 훨씬 더 안전하고 정확합니다.
    loss = (w_variance + h_variance) / image.numel()
    return loss
    # --- [수정 완료] ---

### Additional Loss Functions ###

def compute_gradient_loss(render, gt):
    # X축, Y축 방향의 차분(Gradient) 계산
    def gradient(x):
        # 1픽셀씩 밀어서 차이를 구함 (Finite Difference)
        h_x = x[..., :, 1:] - x[..., :, :-1] # 수평 변화량
        v_x = x[..., 1:, :] - x[..., :-1, :] # 수직 변화량
        return h_x, v_x

    render_dx, render_dy = gradient(render)
    gt_dx, gt_dy = gradient(gt)

    # L1 Loss로 Gradient 차이를 줄임
    loss = torch.abs(render_dx - gt_dx).mean() + torch.abs(render_dy - gt_dy).mean()
    return loss

def compute_fft_loss(render, gt):
    # 1. Fast Fourier Transform (2D)
    fft_render = torch.fft.fft2(render, dim=(-2, -1))
    fft_gt = torch.fft.fft2(gt, dim=(-2, -1))
    
    # 2. 진폭(Amplitude) 스펙트럼 추출
    amp_render = torch.abs(fft_render)
    amp_gt = torch.abs(fft_gt)
    
    # 3. 스펙트럼 간의 L1 거리 계산
    # (옵션: 로그 스케일을 취하면 시각적 인지와 더 비슷해짐)
    return torch.abs(amp_render - amp_gt).mean()



def haar_wavelet_decomposition(x):
    """
    Input: (B, C, H, W) or (C, H, W)
    Output: LL, LH, HL, HH
    """
    # --- [수정된 부분] 차원 체크 및 배치 차원 추가 ---
    if x.ndim == 3:
        x = x.unsqueeze(0)  # (C, H, W) -> (1, C, H, W)
    # ---------------------------------------------

    B, C, H, W = x.shape
    
    # Haar Wavelet Filters
    filters = torch.tensor([
        [[1, 1], [1, 1]],   # LL
        [[1, -1], [1, -1]], # LH
        [[1, 1], [-1, -1]], # HL
        [[1, -1], [-1, 1]]  # HH
    ], dtype=x.dtype, device=x.device).unsqueeze(1) / 4.0 

    # 채널별 연산을 위해 Reshape
    x_reshaped = x.view(B * C, 1, H, W)
    
    # Convolution으로 Wavelet 변환 수행
    out = F.conv2d(x_reshaped, filters, stride=2, groups=1)
    
    # 다시 원본 배치 형태로 복원
    out = out.view(B, C, 4, H // 2, W // 2)
    
    return out[:, :, 0], out[:, :, 1], out[:, :, 2], out[:, :, 3]


class WaveletLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        """
        pred, target: (B, C, H, W) - 렌더링 이미지 및 GT
        edge_dist_map: (B, 1, H, W) - 엣지까지의 거리 (0: 엣지 위, 값이 클수록 엣지에서 멂)
        """
        # 1. Wavelet 분해
        pred_LL, pred_LH, pred_HL, pred_HH = haar_wavelet_decomposition(pred)
        target_LL, target_LH, target_HL, target_HH = haar_wavelet_decomposition(target)
        

        # 4. Loss 계산
        # LL 밴드: 저주파는 그냥 비교 (색상 보존)
        loss_LL = F.l1_loss(pred_LL, target_LL)
        loss_LH = F.l1_loss(pred_LH, target_LH)
        loss_HL = F.l1_loss(pred_HL, target_HL)
        loss_HH = F.l1_loss(pred_HH, target_HH)
        
        
        # 전체 Loss 합산
        total_loss = loss_LL + loss_LH + loss_HL + loss_HH
        
        return total_loss

