import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
from src.losses.relative_camera_loss import (
    quaternion_to_axis_angle,
    quaternion_to_rotation_matrix,
    rotation_matrix_to_quaternion
)

def pseudo_refine(batch, relative_rot_quats, relative_trans):
    """
    Vectorized refinement of pseudo rotations and translations based on relative poses.

    Args:
        batch (dict): Contains:
            - "pseudo_rot": (N, 3, 3) tensor of rotation matrices.
            - "pseudo_trans": (N, 3) tensor of translations.
        relative_rot_quats (Tensor): (N, 4), in (w, x, y, z) format.
        relative_trans (Tensor): (N, 3), relative camera positions.
    """
    pseudo_rot = batch["pseudo_rot"]  # (N, 3, 3)
    pseudo_trans = batch["pseudo_trans"]  # (N, 3)
    device = pseudo_rot.device

    N = relative_rot_quats.shape[0]
    assert N % 2 == 0, "Number of relative quaternions must be even."
    M = N // 2

    # Convert (w,x,y,z) -> (x,y,z,w)
    rel_quats = relative_rot_quats[:, [1, 2, 3, 0]]

    # Indices for pairs
    i_indices = torch.arange(0, N, 2, device=device)
    j_indices = i_indices + 1

    Ri = pseudo_rot[i_indices]  # (M, 3, 3)
    Rj = pseudo_rot[j_indices]
    t_i = pseudo_trans[i_indices]  # (M, 3)
    t_j = pseudo_trans[j_indices]

    Rmi = quaternion_to_rotation_matrix(rel_quats[i_indices])  # (M, 3, 3)
    Rmj = quaternion_to_rotation_matrix(rel_quats[j_indices])
    Rij = Rmj @ torch.linalg.inv(Rmi)

    dRij = torch.linalg.inv(Rj) @ Rij @ Ri
    wij = quaternion_to_axis_angle(rotation_matrix_to_quaternion(dRij))  # (M, 3)
    theta = torch.norm(wij, dim=1, keepdim=True)  # (M, 1)

    valid = theta[:, 0] > 1e-8
    wij_valid = wij[valid]
    Ri_valid = Ri[valid]
    Rj_valid = Rj[valid]

    dRi = axis_angle_to_matrix(-wij_valid / 2)  # (M_valid, 3, 3)
    dRj = axis_angle_to_matrix(wij_valid / 2)
    Ri[valid] = Ri_valid @ dRi
    Rj[valid] = Rj_valid @ dRj

    # Trans update
    c_i = relative_trans[i_indices]  # (M, 3)
    c_j = relative_trans[j_indices]

    r1 = torch.linalg.solve(Rmi, c_i.unsqueeze(-1)).squeeze(-1) - \
         torch.linalg.solve(Rmj, c_j.unsqueeze(-1)).squeeze(-1)  # (M, 3)
    r1_norm = torch.norm(r1, dim=1, keepdim=True)  # (M, 1)

    valid_t = r1_norm > 1e-8
    valid_t = valid_t[:, 0]
    r1_u = torch.zeros_like(r1)
    r1_u[valid_t] = r1[valid_t] / r1_norm[valid_t]

    t0 = torch.linalg.solve(Rmi, t_i.unsqueeze(-1)).squeeze(-1) - \
         torch.linalg.solve(Rmj, t_j.unsqueeze(-1)).squeeze(-1)  # (M, 3)
    t0_norm = torch.norm(t0, dim=1, keepdim=True)

    E = r1_u * t0_norm - t0
    di = E / 2
    dj = -E / 2

    t_i = t_i + di
    t_j = t_j + dj

    # Write back to batch
    pseudo_rot[i_indices] = Ri
    pseudo_rot[j_indices] = Rj
    pseudo_trans[i_indices] = t_i
    pseudo_trans[j_indices] = t_j

    batch["pseudo_rot"] = pseudo_rot
    batch["pseudo_trans"] = pseudo_trans


def axis_angle_to_matrix(axis_angles: torch.Tensor) -> torch.Tensor:
    """
    Convert a batch of axis-angle vectors to rotation matrices.

    Args:
        axis_angles (torch.Tensor): Tensor of shape (N, 3), each row is an axis-angle vector.

    Returns:
        torch.Tensor: Tensor of shape (N, 3, 3) containing rotation matrices.
    """
    theta = torch.norm(axis_angles, dim=1, keepdim=True)  # (N, 1)
    axis = axis_angles / (theta + 1e-8)                   # (N, 3)
    
    x, y, z = axis[:, 0], axis[:, 1], axis[:, 2]
    
    zero = torch.zeros_like(x)
    K = torch.stack([
        torch.stack([zero, -z,    y], dim=1),
        torch.stack([z,    zero, -x], dim=1),
        torch.stack([-y,   x,    zero], dim=1)
    ], dim=1)  # (N, 3, 3)

    eye = torch.eye(3, device=axis_angles.device, dtype=axis_angles.dtype).unsqueeze(0)  # (1, 3, 3)
    theta = theta.view(-1, 1, 1)  # (N, 1, 1)

    R = eye + torch.sin(theta) * K + (1 - torch.cos(theta)) * (K @ K)

    # Handle small angles (theta ≈ 0) with identity matrix
    small_angles = (theta.squeeze(-1).squeeze(-1) < 1e-8)
    if small_angles.any():
        R[small_angles] = eye

    return R