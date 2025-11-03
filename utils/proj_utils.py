# utils/proj_utils.py
import torch
import math

def project_xyz_to_pixels(xyz, cam):
    """
    [최종 수정 버전]
    3DGS 래스터라이저가 C++/CUDA에서 사용하는 것과 동일한
    "focal length (초점 거리)" 기반의 투영 공식을 사용합니다.
    """
    N = xyz.shape[0]
    ones = torch.ones(N, 1, device=xyz.device, dtype=xyz.dtype)
    homo_world = torch.cat([xyz, ones], dim=1)  # (N,4)

    # --- 1. World -> View Space ---
    # 3DGS 래스터라이저와 동일하게 'world_view_transform'을 사용합니다.
    homo_view = homo_world @ cam.world_view_transform  # (N, 4)
    
    # 뷰 공간 좌표 (x, y, z) 추출
    # 3DGS는 카메라가 Z축을 바라보는 표준 뷰 공간을 사용합니다.
    view_x = homo_view[:, 0]
    view_y = homo_view[:, 1]
    view_z = homo_view[:, 2] # (N,)

    # --- 2. View -> 2D Pixel (핵심: 3DGS 공식) ---
    # C++ 래스터라이저 로직: P_x = (v_x * focal_x) / v_z + cx
    
    # (a) FoV(시야각)를 focal length(초점 거리)로 변환
    tanfovx = math.tan(cam.FoVx * 0.5)
    tanfovy = math.tan(cam.FoVy * 0.5)
    focal_x = cam.image_width / (2.0 * tanfovx)
    focal_y = cam.image_height / (2.0 * tanfovy)
    
    # (b) 카메라 중심 (Principal Point)
    # (표준 3DGS는 이미지의 중앙을 사용합니다)
    cx = cam.image_width / 2.0
    cy = cam.image_height / 2.0

    # (c) 투영 공식 적용
    # (view_z로 나누어 원근감을 적용)
    # (focal_y에 -를 붙여 Y축을 뒤집는 것이 일반적일 수 있으나, 
    #  이전 테스트 결과(image_e866c4.png)가 뒤집히지 않은 것을 보면 
    #  3DGS는 +를 사용하는 것으로 보입니다)
    
    # view_z가 0에 가까운 값(카메라 위치)으로 나누는 것을 방지
    view_z = view_z.clamp(min=1e-8) 

    u = (view_x * focal_x) / view_z + cx
    v = (view_y * focal_y) / view_z + cy # (만약 결과가 상하로 뒤집히면 -focal_y로 변경)

    # --- 3. Frustum Culling (카메라 뒤쪽 제거) ---
    # (w > 0 대신, 뷰 공간의 Z좌표 > 0을 확인)
    frustum_mask = (view_z > 1e-3) # (N,)
    
    # --- 4. 최종 'in-bounds' 마스크 ---
    image_mask = (u >= 0) & (u < cam.image_width) & \
                 (v >= 0) & (v < cam.image_height)
    
    inb = frustum_mask & image_mask

    # 5. UV 좌표 반환
    uv = torch.stack([u, v], dim=1)
    
    return uv, inb