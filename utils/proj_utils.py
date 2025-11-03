# utils/proj_utils.py
import torch

def project_xyz_to_pixels(xyz, cam):
    """
    xyz: (N,3) CUDA
    cam.world_view_transform / cam.full_proj_transform: (4,4) CUDA
    반환: uv (N,2), inb (N,) - 픽셀 좌표와 프러스텀 내/이미지 내 마스크
    """
    N = xyz.shape[0]
    ones = torch.ones(N, 1, device=xyz.device, dtype=xyz.dtype)
    homo = torch.cat([xyz, ones], dim=1)  # (N,4)

    # World -> View -> Clip 공간으로 변환
    clip = (homo @ cam.world_view_transform.T) @ cam.full_proj_transform.T  # (N,4)

    # --- 1. 올바른 Frustum Culling ---
    # w 좌표 (clip[:, 3])가 양수여야 카메라 앞에 있는 점입니다.
    # (주의: 씬 구성에 따라 0.0 대신 1e-4 같은 작은 epsilon을 쓰기도 합니다)
    w = clip[:, 3:4]
    frustum_mask = (w > 0.0).squeeze()  # (N,)

    # --- 2. 원근 나누기 (NDC 계산) ---
    ndc = clip[:, :3] / w.clamp(min=1e-8)

    # --- 3. NDC -> Pixel 좌표 ---
    u = (ndc[:, 0] * 0.5 + 0.5) * cam.image_width
    v = (-ndc[:, 1] * 0.5 + 0.5) * cam.image_height  # Y축 뒤집힘

    # --- 4. 이미지 경계 마스크 ---
    # 이 마스크는 u,v가 이미지 내에 있는지 확인합니다.
    image_mask = (u >= 0) & (u < cam.image_width) & \
                 (v >= 0) & (v < cam.image_height)

    # --- 5. 최종 'in-bounds' 마스크 ---
    # (카메라 앞에 있음) AND (이미지 경계 내에 있음)
    # frustum_mask가 (N,), image_mask가 (N,) 이므로 & 연산 가능
    inb = frustum_mask & image_mask

    # 6. UV 좌표 반환 (이전 코드와 동일하게 clamp)
    # 어차피 호출하는 쪽에서 'inb' 마스크로 필터링할 것이므로
    # clamp는 사실상 안전장치 역할만 합니다.
    uv = torch.stack([u.clamp(0, cam.image_width - 1),
                      v.clamp(0, cam.image_height - 1)], dim=1)

    return uv, inb