# utils/edge_utils.py
import cv2, torch, numpy as np
import torch.nn.functional as F
import torchvision.transforms.functional as TF

def build_edge_distance_map(rgb_tensor_chw_01):
    # rgb: (3,H,W) [0,1] torch
    img = (rgb_tensor_chw_01.permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # edges = cv2.Canny(gray, 100, 200)
    edges = cv2.Canny(gray, 10, 200)
    inv = (edges == 0).astype(np.uint8)
    dist = cv2.distanceTransform(inv, cv2.DIST_L2, 3).astype(np.float32)
    if dist.max() > 0:
        dist /= (dist.max() + 1e-6)  # [0,1]
    return torch.from_numpy(dist)  # (H,W) torch

# def build_texture_distance_map(rgb_tensor_chw_01, texture_threshold=0.15):
#     """
#     텍스처 에너지 맵을 '거리 맵' 형태로 변환하여 반환하는 함수.
#     - texture_threshold: 텍스처로 간주할 에너지의 최소값 (0~1)
#       이 값이 낮을수록 미세한 텍스처도 '구조'로 인식하여 주변을 보호함.
#     """
#     # 1. 기존 Texture Energy Map 계산 (0:텍스처 ~ 1:평지)
#     # (내부적으로 build_texture_energy_map을 호출한다고 가정)
#     # 결과가 Tensor라면 Numpy로 변환 필요
#     energy_map = build_texture_energy_map(rgb_tensor_chw_01, kernel_size=3)
    
#     # GPU Tensor -> CPU Numpy 변환
#     if isinstance(energy_map, torch.Tensor):
#         energy_map_np = energy_map.detach().cpu().numpy()
#     else:
#         energy_map_np = energy_map

#     # 2. 이진화 (Binarization)
#     # energy_map은 0이 텍스처(중요), 1이 평지(안중요)라고 가정했었음.
#     # 따라서 값이 'threshold'보다 작은 곳이 텍스처 영역임.
#     # 텍스처인 곳 = 0 (Black), 배경 = 1 (White)로 만들어야 Distance Transform이 됨.
    
#     # 텍스처가 있는 곳을 0으로, 없는 곳을 1로 만듦
#     binary_mask = (energy_map_np > texture_threshold).astype(np.uint8)

#     # 3. Distance Transform 수행
#     # binary_mask에서 '0'인 픽셀(텍스처)까지의 거리를 계산
#     dist = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 3).astype(np.float32)

#     # 4. 정규화 [0, 1]
#     if dist.max() > 0:
#         # 거리가 멀어질수록 1.0에 가까워짐 (평평함)
#         # 거리가 가까울수록 0.0에 가까워짐 (텍스처 근처)
#         dist /= (dist.max() + 1e-6)

#     # 5. 다시 Tensor로 변환
#     return torch.from_numpy(dist).to(rgb_tensor_chw_01.device)

# def build_soft_texture_energy_map(rgb_tensor_chw_01, kernel_size=3, sensitivity_gain=20.0, smooth_sigma=1.0):
#     """
#     부드럽고 관대한 Texture Energy Map 생성
    
#     Args:
#         sensitivity_gain: 텍스처 신호 증폭값. 클수록 미세한 텍스처도 중요하게(0.0) 잡음.
#                           (기존 max_texture_val=0.1은 gain=10.0과 유사)
#                           추천값: 15.0 ~ 30.0
#         smooth_sigma: 맵의 경계를 부드럽게 할 Gaussian Blur의 sigma 값.
#                       0이면 기존처럼 Strict함. 클수록 맵이 몽글몽글해짐.
#     """
#     # 1. Grayscale 변환
#     x = rgb_tensor_chw_01.unsqueeze(0) 
#     gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]

#     # 2. Local Variance 계산 (기존과 동일)
#     padding = kernel_size // 2
#     mean = F.avg_pool2d(gray, kernel_size=kernel_size, stride=1, padding=padding)
#     mean_sq = F.avg_pool2d(gray**2, kernel_size=kernel_size, stride=1, padding=padding)
#     variance = torch.clamp(mean_sq - mean**2, min=0)
#     std = torch.sqrt(variance).squeeze() # (H, W)

#     # --- [핵심 개선 1] 비선형 매핑 (Tanh 활성화 함수 사용) ---
#     # 기존: linear_clamp = clamp(std / 0.1) -> 0.05는 0.5(애매함)가 됨
#     # 개선: tanh = tanh(std * gain)         -> 0.05 * 20 = 1.0 -> tanh(1.0) = 0.76 (꽤 중요해짐)
#     #
#     # Tanh 특성상 작은 값은 급격하게 커지고, 큰 값은 1.0에 부드럽게 수렴합니다.
#     # 따라서 "회색 지대"의 텍스처들이 "검정색(중요)" 쪽으로 확 살아납니다.
#     texture_strength = torch.tanh(std * sensitivity_gain)

#     # --- [핵심 개선 2] 공간적 스무딩 (Gaussian Blur) ---
#     # 픽셀 하나하나가 튀는 것을 방지하고, 텍스처 영역을 주변으로 "번지게" 합니다.
#     # Strict한 경계를 허물어줍니다.
#     if smooth_sigma > 0:
#         # (H, W) -> (1, 1, H, W) for blurring
#         texture_strength = texture_strength.unsqueeze(0).unsqueeze(0)
#         kernel_size_blur = int(2 * round(4 * smooth_sigma) + 1) # sigma에 맞는 커널 크기 자동 계산
#         texture_strength = TF.gaussian_blur(texture_strength, kernel_size=kernel_size_blur, sigma=smooth_sigma)
#         texture_strength = texture_strength.squeeze()

#     # 4. 반전 (0=Texture/Important, 1=Flat)
#     final_map = 1.0 - texture_strength

#     return final_map


# def build_texture_energy_map(rgb_tensor_chw_01, kernel_size=3):
#     """
#     로컬 분산(Variance)을 기반으로 텍스처 맵 생성
#     - rgb_tensor_chw_01: (3, H, W) tensor, range [0, 1]
#     - return: (H, W) tensor, range [0, 1]
#       * 값 0.0에 가까울수록: 텍스처가 많고 복잡함 (보존해야 함)
#       * 값 1.0에 가까울수록: 완전히 평평함 (Pruning 해도 됨)
#     """
#     # 1. Grayscale 변환 (가중치 사용)
#     # (B, C, H, W) 형태로 맞추기 위해 unsqueeze
#     x = rgb_tensor_chw_01.unsqueeze(0) 
#     gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]

#     # 2. 로컬 평균 (E[X]) 계산
#     # padding을 해서 원본 크기 유지
#     padding = kernel_size // 2
#     mean = F.avg_pool2d(gray, kernel_size=kernel_size, stride=1, padding=padding)

#     # 3. 로컬 제곱의 평균 (E[X^2]) 계산
#     mean_sq = F.avg_pool2d(gray**2, kernel_size=kernel_size, stride=1, padding=padding)

#     # 4. 분산(Variance) 계산: Var = E[X^2] - (E[X])^2
#     variance = mean_sq - mean**2
    
#     # 수치적 안정성을 위해 음수 방지
#     variance = torch.clamp(variance, min=0)

#     # 5. 표준편차(Std)로 변환 (선형적인 척도로 만듦)
#     std = torch.sqrt(variance).squeeze() # (H, W)

#     # 6. 정규화 및 반전 (Inversion)
#     # 기존 로직(distance map)과 호환되게 하기 위해:
#     # 텍스처가 많음 -> std 높음 -> 값을 0에 가깝게 매핑
#     # 텍스처가 없음 -> std 낮음 -> 값을 1에 가깝게 매핑
    
#     # 텍스처 강도 상한선 설정 (이 값 이상이면 '완전한 텍스처'로 간주)
#     # 보통 0.1~0.2 정도면 꽤 강한 텍스처임
#     max_texture_val = 0.1 
    
#     normalized_texture = torch.clamp(std / max_texture_val, 0, 1)
    
#     # 반전: 0(Texture/Edge) ~ 1(Flat)
#     final_map = 1.0 - normalized_texture

#     return final_map

# def build_hybrid_importance_map(rgb_tensor_chw_01, texture_sensitivity=1.0):
#     """
#     Edge Distance Map(구조)과 Texture Energy Map(디테일)을 결합
#     - texture_sensitivity: 텍스처 맵의 강도를 조절 (0.0 ~ 1.0)
#         * 1.0: 텍스처를 엣지만큼 중요하게 생각함 (0.0까지 내려감)
#         * 0.5: 텍스처는 '적당히' 중요하게 생각함 (0.5까지만 내려감)
#     """
    
#     # 1. 기존 Edge Distance Map 계산 (구조 담당)
#     # (함수 호출 시 내부에서 CPU/GPU 변환 등을 주의하세요)
#     map_edge = build_edge_distance_map(rgb_tensor_chw_01)
#     if map_edge.device != rgb_tensor_chw_01.device:
#         map_edge = map_edge.to(rgb_tensor_chw_01.device)

#     # 2. Texture Energy Map 계산 (디테일 담당)
#     map_texture = build_texture_energy_map(rgb_tensor_chw_01, kernel_size=3)
#     if map_texture.device != rgb_tensor_chw_01.device:
#         map_texture = map_texture.to(rgb_tensor_chw_01.device)

#     # # [옵션] 텍스처 민감도 조절
#     # # 텍스처 맵이 너무 공격적(너무 많이 살림)이라면 강도를 줄일 수 있습니다.
#     # # 예: sensitivity=0.5면, 텍스처 맵의 최소값이 0.5가 되어 "절대 보호"는 안 됨
#     # if texture_sensitivity < 1.0:
#     #     # 0(중요) ~ 1(안중요) 범위를 -> (1-sens) ~ 1 범위로 압축
#     #     # 텍스처가 아무리 강해도 0이 되지 않고 (1-sens) 까지만 내려감
#     #     map_texture = 1.0 - (1.0 - map_texture) * texture_sensitivity

#     # # 3. 결합 (Minimum Fusion)
#     # # "둘 중 더 중요한(값이 작은) 쪽을 따른다"
#     # map_hybrid = torch.min(map_edge, map_texture)

#     ## 단순하게 더하는 방식으로 결합
#     map_hybrid = map_edge + map_texture

#     # 정규화: [0, 2] -> [0, 1]
#     map_hybrid = map_hybrid / 2.0

#     return map_hybrid


# def build_seamless_hybrid_importance_map(rgb_tensor_chw_01, 
#                                           texture_weight=0.5, # 텍스처 맵의 기여도 조절
#                                           edge_power=1.0,     # 엣지 맵의 '퍼짐' 정도 조절
#                                           texture_power=1.0,  # 텍스처 맵의 '강도' 조절
#                                           min_importance=0.0): # 맵의 최소 중요도 (너무 흰색되는 것 방지)
#     """
#     Edge Distance Map(구조)과 Texture Energy Map(디테일)을 계층적 가중치 방식으로 결합
#     - rgb_tensor_chw_01: (3, H, W) tensor, range [0, 1]
#     - texture_weight: texture map이 최종 맵에 얼마나 기여할지 (0.0 ~ 1.0)
#     - edge_power: 엣지 맵의 값을 power 함수로 조정 (1.0 = 선형, <1.0 = 더 넓게 퍼짐)
#     - texture_power: 텍스처 맵의 값을 power 함수로 조정 (1.0 = 선형, >1.0 = 더 강조)
#     - min_importance: 최종 맵의 최소 중요도. (0.0 = 완전 흰색 허용, 0.1 = 연회색까지만 허용)
#     """
    
#     # 1. Edge Distance Map 계산 (구조 담당)
#     map_edge = build_edge_distance_map(rgb_tensor_chw_01)
#     if map_edge.device != rgb_tensor_chw_01.device:
#         map_edge = map_edge.to(rgb_tensor_chw_01.device)

#     # 2. Texture Energy Map 계산 (디테일 담당)
#     map_texture = build_texture_energy_map(rgb_tensor_chw_01, kernel_size=3)
#     if map_texture.device != rgb_tensor_chw_01.device:
#         map_texture = map_texture.to(rgb_tensor_chw_01.device)
    
#     # --- 핵심 결합 로직 ---

#     # 1단계: 엣지 맵 조정 (strict하지 않게, 더 넓게 퍼지도록)
#     # map_edge가 0에 가까울수록(중요) 더 큰 변화를 줌.
#     # edge_power < 1.0 이면 맵이 전반적으로 더 어두워지고, '퍼지는 효과'가 강해집니다.
#     # 예: 0.5 -> 맵 값이 0.1은 0.31로, 0.5는 0.707로, 0.9는 0.94로. 0에 가까운 값들이 더 커지면서 보호 영역이 넓어짐.
#     adjusted_edge_map = torch.pow(map_edge, edge_power)


#     # 2단계: 텍스처 맵 조정 (디테일 강조)
#     # map_texture가 0에 가까울수록(중요) 더 큰 변화를 줌.
#     # texture_power > 1.0 이면 맵이 전반적으로 더 어두워지고, 디테일이 강조됩니다.
#     adjusted_texture_map = torch.pow(map_texture, texture_power)

    
#     # 3단계: Weighted Sum (가중치 합산)
#     # 엣지 맵에 더 큰 가중치를 줘서 기본 구조를 담당하게 하고,
#     # 텍스처 맵은 '추가적인 디테일'을 더하는 방식으로 결합합니다.
#     # 이 부분에서 'seamless' 함이 결정됩니다.
#     # (1 - texture_weight) 만큼 엣지맵이 기여하고, texture_weight 만큼 텍스처맵이 기여.
#     # 결과 값은 [0, 1] 범위를 넘어설 수 있으므로, 최종 정규화 필요
#     weighted_sum_map = (1.0 - texture_weight) * adjusted_edge_map + texture_weight * adjusted_texture_map
    

#     # 4단계: 최종 정규화 및 클램핑
#     # 맵 값이 [0, 1] 범위에 들어오도록 클램프. (weighted_sum_map은 0~1 사이)
#     # min_importance: 맵의 최소 중요도를 설정하여 너무 하얗게 되는 것을 방지
#     # (0.0은 완전 평지, 0.1은 연한 회색)
#     final_hybrid_map = torch.clamp(weighted_sum_map, min=min_importance, max=1.0)

#     return final_hybrid_map


# def build_multires_edge_distance_map(rgb_tensor_chw_01):
#     """
#     Multi-resolution Edge Distance Map 생성 함수
#     - rgb_tensor_chw_01: (3, H, W) tensor, range [0, 1]
#     - return: (H, W) tensor, range [0, 1] (값이 작을수록 엣지/텍스처에 가까움)
#     """
#     # 1. 이미지 전처리 (Tensor -> Numpy Grayscale)
#     img = (rgb_tensor_chw_01.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
#     gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#     h, w = gray.shape

#     # ---------------------------------------------------------
#     # [Level 0] High Resolution (Original) - 미세 텍스처 포착
#     # ---------------------------------------------------------
#     # threshold를 낮게(50, 150) 설정하여 자글자글한 텍스처와 약한 엣지를 잡습니다.
#     edges_high = cv2.Canny(gray, 50, 150)

#     # ---------------------------------------------------------
#     # [Level 1] Middle Resolution (1/2 Downsample) - 중간 구조 포착
#     # ---------------------------------------------------------
#     gray_small = cv2.pyrDown(gray) # Gaussian Blur + Downsample
#     # threshold를 표준(100, 200)으로 설정하여 노이즈를 제외한 구조를 잡습니다.
#     edges_mid_raw = cv2.Canny(gray_small, 100, 200)
#     # 원본 크기로 복원 (Nearest Neighbor가 엣지 보존에 유리할 수 있으나, Linear도 무방)
#     edges_mid = cv2.resize(edges_mid_raw, (w, h), interpolation=cv2.INTER_NEAREST)

#     # ---------------------------------------------------------
#     # [Level 2] Low Resolution (1/4 Downsample) - 큰 윤곽선(Dominant) 포착
#     # ---------------------------------------------------------
#     gray_tiny = cv2.pyrDown(gray_small)
#     # 매우 강한 엣지만 잡기 위해 threshold 유지
#     edges_low_raw = cv2.Canny(gray_tiny, 100, 200)
#     edges_low = cv2.resize(edges_low_raw, (w, h), interpolation=cv2.INTER_NEAREST)

#     # ---------------------------------------------------------
#     # [Union] 엣지 통합 (Texture + Structure)
#     # ---------------------------------------------------------
#     # 모든 스케일의 엣지를 합칩니다 (OR 연산).
#     # 하나라도 엣지로 감지된 픽셀은 255(Edge)가 됩니다.
#     combined_edges = cv2.bitwise_or(edges_high, edges_mid)
#     combined_edges = cv2.bitwise_or(combined_edges, edges_low)

#     # ---------------------------------------------------------
#     # [Distance Transform] 거리 맵 생성
#     # ---------------------------------------------------------
#     # 엣지인 부분(255)은 0, 배경(0)은 1로 반전
#     inv = (combined_edges == 0).astype(np.uint8)
    
#     # L2 Distance 계산 (가장 가까운 엣지까지의 거리)
#     # 결과적으로 텍스처가 많은 영역은 엣지가 빽빽하므로 거리가 0에 가깝게 나옵니다.
#     dist = cv2.distanceTransform(inv, cv2.DIST_L2, 3).astype(np.float32)

#     # 정규화 [0, 1]
#     if dist.max() > 0:
#         # 평평한 영역일수록 값이 1.0에 가깝고, 텍스처/엣지는 0.0에 가까움
#         dist /= (dist.max() + 1e-6)

#     return torch.from_numpy(dist) # (H, W) Torch Tensor



def soft_edge_map(pred_bchw_01, kappa=4.0, eps=1e-6):
    # pred_bchw_01: [B,3,H,W] in [0,1]
    xg = pred_bchw_01.mean(1, keepdim=True)                 # [B,1,H,W]
    sobel_x = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], device=xg.device, dtype=xg.dtype).view(1,1,3,3)/8.0
    sobel_y = sobel_x.transpose(-1,-2)
    gx = F.conv2d(xg, sobel_x, padding=1)
    gy = F.conv2d(xg, sobel_y, padding=1)
    mag = (gx*gx + gy*gy + eps).sqrt()
    mag = mag / (mag.amax(dim=(2,3), keepdim=True) + eps)   # [0,1]
    return torch.sigmoid(kappa*(mag - 0.5))                 # [B,1,H,W]



# def build_unified_edge_texture_map(rgb_tensor_chw_01, texture_threshold=0.15):
#     """
#     구조적 엣지와 텍스처를 모두 '엣지'로 취급하여 통합 거리 맵을 생성
#     - texture_threshold: 텍스처 에너지 맵에서 이 값 이상이면 '엣지'로 간주함.
#       (값을 낮출수록 미세한 텍스처도 엣지처럼 강력하게 보호됨)
#     """
#     # 1. 이미지 준비 (Tensor -> Numpy Grayscale)
#     img_np = (rgb_tensor_chw_01.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
#     gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
#     h, w = gray.shape

#     # ----------------------------------------------------
#     # A. Structural Edges (기존 방식 - 큰 구조)
#     # ----------------------------------------------------
#     # Canny로 강한 윤곽선 추출 (값이 255인 픽셀이 엣지)
#     edges_structure = cv2.Canny(gray, 100, 200)

#     # ----------------------------------------------------
#     # B. Texture Edges (새로운 방식 - 미세 패턴)
#     # ----------------------------------------------------
#     # 1. Texture Energy 계산 (Local Variance)
#     # (함수 내부 로직을 간단히 구현)
#     x = rgb_tensor_chw_01.unsqueeze(0) # (1, 3, H, W)
#     gray_tensor = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
    
#     # Variance 계산 (Kernel size 3)
#     padding = 1
#     mean = F.avg_pool2d(gray_tensor, 3, stride=1, padding=padding)
#     mean_sq = F.avg_pool2d(gray_tensor**2, 3, stride=1, padding=padding)
#     variance = torch.clamp(mean_sq - mean**2, min=0)
#     std = torch.sqrt(variance).squeeze().cpu().numpy() # (H, W) Numpy

#     # 2. Thresholding: 에너지가 높은 곳을 '엣지(255)'로 변환
#     # texture_threshold(예: 0.15)보다 표준편차가 크면 엣지로 봅니다.
#     edges_texture = np.zeros_like(std, dtype=np.uint8)
#     edges_texture[std > texture_threshold] = 255

#     # ----------------------------------------------------
#     # C. Union & Distance Transform (핵심)
#     # ----------------------------------------------------
#     # 1. 두 엣지 마스크 합치기 (OR 연산)
#     # 이제 굵은 윤곽선도 255, 자글자글한 바닥 무늬도 255가 됩니다.
#     combined_edges = cv2.bitwise_or(edges_structure, edges_texture)

#     # 2. Distance Transform 수행
#     # 엣지(255)는 거리가 0, 멀어질수록 값이 커짐
#     # 반전 필요: Distance Transform은 0(검정)이 배경, 1(흰색)이 물체일 때 물체까지의 거리를 잼
#     # 여기서는 엣지까지의 거리를 재야 하므로 엣지를 0으로 만들어야 함.
#     # cv2.distanceTransform은 '0이 아닌 픽셀'에서 '0인 픽셀'까지의 거리를 잰다.
#     # 따라서 엣지를 0으로, 배경을 1로 만든 이미지를 입력으로 줘야 함.
    
#     dist_input = np.ones_like(combined_edges, dtype=np.uint8)
#     dist_input[combined_edges > 0] = 0 # 엣지인 곳을 0으로 설정

#     # 거리 계산 (L2 Distance)
#     dist_map = cv2.distanceTransform(dist_input, cv2.DIST_L2, 3).astype(np.float32)

#     # 3. 정규화 [0, 1]
#     # 거리가 0(엣지 위) ~ 1(평지)
#     if dist_map.max() > 0:
#         dist_map /= (dist_map.max() + 1e-6)

#     # Tensor로 변환 및 디바이스 이동
#     return torch.from_numpy(dist_map).to(rgb_tensor_chw_01.device)


def build_texture_based_distance_map(rgb_tensor_chw_01, kernel_size=3, texture_threshold=0.025): # 0.025
    """
    모든 텍스처 영역을 '엣지'로 간주하고 Distance Map을 생성하는 함수
    
    Args:
        texture_threshold: 표준편차(std)가 이 값보다 크면 '텍스처 엣지'로 간주.
                           낮을수록(예: 0.02) 미세한 노이즈도 엣지로 쳐서 보호 범위가 넓어짐.
                           높을수록(예: 0.10) 확실한 무늬만 엣지로 침.
    Returns:
        (H, W) Tensor [0, 1] (0에 가까울수록 텍스처 근처, 1은 완전 평지)
    """
    
    # --- [Step 1] Texture Energy (Std) 계산 (기존과 동일) ---
    # (GPU 연산 활용)
    x = rgb_tensor_chw_01.unsqueeze(0) 
    gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]

    padding = kernel_size // 2
    mean = F.avg_pool2d(gray, kernel_size=kernel_size, stride=1, padding=padding)
    mean_sq = F.avg_pool2d(gray**2, kernel_size=kernel_size, stride=1, padding=padding)
    variance = torch.clamp(mean_sq - mean**2, min=0)
    std = torch.sqrt(variance).squeeze() # (H, W) Tensor

    # --- [Step 2] Thresholding (텍스처를 엣지로 변환) ---
    # GPU Tensor -> CPU Numpy 변환 (cv2 사용을 위해)
    std_np = std.detach().cpu().numpy()
    
    # 텍스처가 있는 곳(std > threshold)을 '0' (Distance Transform의 목표점)으로 설정
    # 텍스처가 없는 곳을 '1'로 설정
    # (cv2.distanceTransform은 0이 아닌 픽셀에서 0인 픽셀까지의 거리를 잰다)
    binary_mask = np.ones_like(std_np, dtype=np.uint8)
    binary_mask[std_np > texture_threshold] = 0  # 텍스처 발견! -> 여기가 엣지다(0)

    # --- [Step 3] Distance Transform (거리 맵 생성) ---
    # 텍스처 픽셀(0)로부터의 거리를 계산
    dist = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 3).astype(np.float32)

    # --- [Step 4] 정규화 ---
    if dist.max() > 0:
        dist /= (dist.max() + 1e-6) # [0, 1] 정규화

    # 다시 GPU Tensor로 변환
    return torch.from_numpy(dist).to(rgb_tensor_chw_01.device)


def get_otsu_threshold(values):
    """
    PyTorch 텐서(0~1 범위)에 대해 Otsu Threshold를 계산하는 함수
    """
    # 1. 0~1 값을 0~255 정수 bin으로 변환
    values = (values * 255).long().clamp(0, 255)
    
    # 2. 히스토그램 계산
    hist = torch.histc(values.float(), bins=256, min=0, max=255)
    total = values.numel()
    
    current_max, threshold = 0, 0
    sum_total, sum_b, weight_b = 0, 0, 0
    
    # 전체 합 미리 계산
    for i in range(256):
        sum_total += i * hist[i]
        
    # 3. 최적의 임계값 탐색
    for i in range(256):
        weight_b += hist[i]
        if weight_b == 0: continue
        
        weight_f = total - weight_b
        if weight_f == 0: break
        
        sum_b += i * hist[i]
        mean_b = sum_b / weight_b
        mean_f = (sum_total - sum_b) / weight_f
        
        # 클래스 간 분산(Between Class Variance) 계산
        var_between = weight_b * weight_f * (mean_b - mean_f)**2
        
        if var_between > current_max:
            current_max = var_between
            threshold = i
            
    return threshold / 255.0 # 다시 0~1 범위로 복귀