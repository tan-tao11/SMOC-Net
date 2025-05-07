import torch
import torch.nn as nn
from scipy.spatial.transform import Rotation as R


def quaternion_to_rotation_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert a batch of quaternions to corresponding rotation matrices.

    Args:
        quaternions (torch.Tensor): A tensor of shape (..., 4) representing
            a batch of quaternions. The last dimension must contain the
            quaternion components. Quaternions are in (X, Y, Z, W) format.

    Returns:
        torch.Tensor: A tensor of shape (..., 3, 3) representing the
            corresponding rotation matrices for each quaternion in the batch.
    """
    x, y, z, w = quaternions.unbind(dim=-1)

    # Normalize the quaternion to avoid scaling issues
    norm = torch.sqrt(x*x + y*y + z*z + w*w)
    x = x / norm
    y = y / norm
    z = z / norm
    w = w / norm

    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z

    rot = torch.stack([
        1 - 2*(yy + zz), 2*(xy - wz),     2*(xz + wy),
        2*(xy + wz),     1 - 2*(xx + zz), 2*(yz - wx),
        2*(xz - wy),     2*(yz + wx),     1 - 2*(xx + yy)
    ], dim=-1)

    return rot.reshape(quaternions.shape[:-1] + (3, 3))

def rotation_matrix_to_quaternion(rotation_matrices: torch.Tensor) -> torch.Tensor:
    """
    Convert a batch of rotation matrices to corresponding quaternions.

    Args:
        rotation_matrices (torch.Tensor): A tensor of shape (..., 3, 3) representing
            a batch of rotation matrices.

    Returns:
        torch.Tensor: A tensor of shape (..., 4) representing the corresponding
            quaternions in (X, Y, Z, W) format.
    """
    R = rotation_matrices
    batch_shape = R.shape[:-2]

    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    eps = 1e-6

    def compute_q(trace, R):
        q = torch.zeros(*batch_shape, 4, dtype=R.dtype, device=R.device)

        cond1 = trace > 0
        s1 = torch.sqrt(trace[cond1] + 1.0) * 2
        q[cond1, 3] = 0.25 * s1
        q[cond1, 0] = (R[cond1, 2, 1] - R[cond1, 1, 2]) / s1
        q[cond1, 1] = (R[cond1, 0, 2] - R[cond1, 2, 0]) / s1
        q[cond1, 2] = (R[cond1, 1, 0] - R[cond1, 0, 1]) / s1

        cond2 = (R[..., 0, 0] > R[..., 1, 1]) & (R[..., 0, 0] > R[..., 2, 2]) & ~cond1
        s2 = torch.sqrt(1.0 + R[cond2, 0, 0] - R[cond2, 1, 1] - R[cond2, 2, 2]) * 2
        q[cond2, 3] = (R[cond2, 2, 1] - R[cond2, 1, 2]) / s2
        q[cond2, 0] = 0.25 * s2
        q[cond2, 1] = (R[cond2, 0, 1] + R[cond2, 1, 0]) / s2
        q[cond2, 2] = (R[cond2, 0, 2] + R[cond2, 2, 0]) / s2

        cond3 = (R[..., 1, 1] > R[..., 2, 2]) & ~cond1 & ~cond2
        s3 = torch.sqrt(1.0 + R[cond3, 1, 1] - R[cond3, 0, 0] - R[cond3, 2, 2]) * 2
        q[cond3, 3] = (R[cond3, 0, 2] - R[cond3, 2, 0]) / s3
        q[cond3, 0] = (R[cond3, 0, 1] + R[cond3, 1, 0]) / s3
        q[cond3, 1] = 0.25 * s3
        q[cond3, 2] = (R[cond3, 1, 2] + R[cond3, 2, 1]) / s3

        cond4 = ~cond1 & ~cond2 & ~cond3
        s4 = torch.sqrt(1.0 + R[cond4, 2, 2] - R[cond4, 0, 0] - R[cond4, 1, 1]) * 2
        q[cond4, 3] = (R[cond4, 1, 0] - R[cond4, 0, 1]) / s4
        q[cond4, 0] = (R[cond4, 0, 2] + R[cond4, 2, 0]) / s4
        q[cond4, 1] = (R[cond4, 1, 2] + R[cond4, 2, 1]) / s4
        q[cond4, 2] = 0.25 * s4

        return q

    quaternions = compute_q(trace, R)

    # Normalize to ensure unit quaternion
    quaternions = quaternions / quaternions.norm(dim=-1, keepdim=True)

    return quaternions

def quaternion_to_axis_angle(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to axis/angle.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    """
    # Unpack quaternion components
    x, y, z, w = quaternions.unbind(dim=-1)
    
    # Normalize to ensure it's a valid quaternion
    norm = torch.sqrt(w**2 + x**2 + y**2 + z**2)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm

    angle = 2 * torch.acos(torch.clamp(w, -1.0, 1.0))  # Clamp for numerical stability

    # Compute sin(theta/2), use it to get axis direction
    s = torch.sqrt(1 - w**2)
    
    # Avoid division by zero by setting small s to 1 and zeroing out axis
    small_angle = s < 1e-8
    axis = torch.stack([x, y, z], dim=-1)
    axis = torch.where(small_angle[..., None], torch.zeros_like(axis), axis / s[..., None])

    # Return axis scaled by angle
    return axis * angle[..., None]

def compute_rotation_matrix_loss(rot1, rot2):
    axis_angle1 = quaternion_to_axis_angle(rotation_matrix_to_quaternion(rot1))
    axis_angle2 = quaternion_to_axis_angle(rotation_matrix_to_quaternion(rot2))
    axis_angle_diff = axis_angle1 - axis_angle2

    return torch.norm(axis_angle_diff, dim=-1).mean()

class loss_relative_rot(nn.Module):
    def __init__(self, config):
        super(loss_relative_rot, self).__init__()
        self.config = config
        
    def forward(self, relative_rot_quats, pred_rots, num_samples=None, sym_infos=None):
        loss_dict = {}

        if sym_infos[0] is not None:
          sym = True
        else:
            sym = False
        if sym:
            sym_info = torch.from_numpy(sym_infos[0]).to(torch.device('cuda:0')).to(torch.float32)
        # (w, x, y, z) -> (x, y, z, w)
        relative_rot_quats_ = relative_rot_quats[:, [1, 2, 3, 0]]

        if not sym:
            # Quaternion to rotation matrix
            relative_mats = quaternion_to_rotation_matrix(relative_rot_quats_)
            pred_mats = pred_rots

            # Reshape (B, 3, 3) to (B/2, 2, 3, 3)
            relative_pairs = relative_mats.view(-1, 2, 3, 3)
            pred_pairs = pred_mats.view(-1, 2, 3, 3)

            relative_mat_i, relative_mat_j = relative_pairs.unbind(dim=1)
            pred_mat_i, pred_mat_j = pred_pairs.unbind(dim=1)

            # Compute the relative rotation matrix
            relative_rot_mats = torch.einsum('bij,bjk->bik', relative_mat_i, relative_mat_j.transpose(1, 2))
            pred_rot_mats = torch.einsum('bij,bjk->bik', pred_mat_i, pred_mat_j.transpose(1, 2))

            # Compute the loss
            loss = compute_rotation_matrix_loss(relative_rot_mats, pred_rot_mats)
            loss_dict['loss_relative_rot'] = loss * self.config.loss_weight
        else:
            losses = []
            for i in range(0, len(relative_rot_quats_), 2):
                # Quaternion to rotation matrix
                relative_mat_i = quaternion_to_rotation_matrix(relative_rot_quats_[i])
                relative_mat_j = quaternion_to_rotation_matrix(relative_rot_quats_[i+1])

                # Compute the relative rotation matrix
                relative_rot_mat = torch.matmul(relative_mat_i, relative_mat_j.transpose(0, 1))

                pred_mat_i = pred_rots[i]
                pred_mat_j = pred_rots[i+1]
                
                # Compute the predicted relative rotation matrix
                pred_rot_mat = torch.matmul(pred_mat_i, pred_mat_j.transpose(0, 1))

                # Generate symmetric matrix
                relative_mat_i = relative_mat_i.expand(len(sym_info), 3, 3)
                relative_mat_i_sym = torch.bmm(relative_mat_i, sym_info)
                relative_mat_i_sym_b = relative_mat_i_sym.expand(len(sym_info),len(sym_info), 3, 3)
                relative_mat_i_sym_b = torch.reshape(relative_mat_i_sym_b, [-1, 3, 3])

                relative_mat_j_sym = relative_mat_j_sym.expand(len(sym_info), 3, 3)
                relative_mat_j_sym = torch.bmm(relative_mat_j, sym_info)
                relative_mat_j_sym_b = relative_mat_j_sym.expand(len(sym_info),len(sym_info), 3, 3)
                relative_mat_j_sym_b = torch.reshape(relative_mat_j_sym_b, [-1, 3, 3])
                relative_mat_j_sym_b_inv = relative_mat_j_sym_b.transpose(1,2)

                relative_rot_mat_sym = torch.bmm(relative_mat_i_sym_b, relative_mat_j_sym_b_inv)

                pred_rot_mat_sym = pred_rot_mat.expand(len(relative_rot_mat_sym), 3, 3)

                loss = compute_rotation_matrix_loss(relative_rot_mat_sym, pred_rot_mat_sym)
                losses.append(loss)
            loss = torch.mean(torch.stack(losses))
            loss_dict['loss_relative_rot'] = loss * self.config.loss_weight

        return loss_dict
    
class loss_relative_trans(nn.Module):
    def __init__(self, config):
        super(loss_relative_trans, self).__init__()
        self.config = config

    def forward(self, relative_trans, relative_rot_quats, pred_trans):
        loss_dict = {}
        # (w, x, y, z) -> (x, y, z, w)
        relative_rot_quats_ = relative_rot_quats[:, [1, 2, 3, 0]]
        # Quaternion to rotation matrix
        relative_mats = quaternion_to_rotation_matrix(relative_rot_quats_)

        relative_trans = relative_trans.view(-1, 3)
        pred_trans = pred_trans.view(-1, 3)
        relative_mats = relative_mats.view(-1, 3, 3)

        i_indices = torch.arange(0, relative_trans.shape[0], 2)
        j_indices = i_indices + 1

        mat_i = relative_mats[i_indices]
        mat_j = relative_mats[j_indices]
        trans_i = relative_trans[i_indices]
        trans_j = relative_trans[j_indices]
        pred_i = pred_trans[i_indices]
        pred_j = pred_trans[j_indices]

        # Compute the ground-truth relative translation vector
        gt_ij = torch.einsum("bij,bj->bi", mat_i.transpose(1, 2), trans_i) - \
                    torch.einsum("bij,bj->bi", mat_j.transpose(1, 2), trans_j)
        gt_ij_norm = torch.norm(gt_ij, dim=1, keepdim=True)
        valid_mask = gt_ij_norm.squeeze(-1) > 0
        # gt_ij = gt_ij[valid_mask] / gt_ij_norm[valid_mask]
        gt_ij = gt_ij / gt_ij_norm

        # Compute the predicted relative translation vector
        pred_ij = torch.einsum("bij,bj->bi", mat_i.transpose(1, 2), pred_i) - \
        torch.einsum("bij,bj->bi", mat_j.transpose(1, 2), pred_j)
        pred_ij_norm = torch.norm(pred_ij, dim=1, keepdim=True)
        valid_mask_pred = pred_ij_norm.squeeze(-1) > 0
        # pred_ij = pred_ij[valid_mask_pred] / pred_ij_norm[valid_mask_pred]
        pred_ij = pred_ij / pred_ij_norm

        final_mask = valid_mask & valid_mask_pred

        gt_ij = gt_ij[final_mask]
        pred_ij = pred_ij[final_mask]
        loss = torch.norm(gt_ij - pred_ij, dim=1).mean()

        loss_dict["loss_relative_trans"] = loss * self.config.loss_weight
        return loss_dict
        # for i in range(0, len(relative_trans)/2, 2):
        #     relative_mat_i = relative_mats[i]
        #     relative_mat_j = relative_mats[i+1]

        #     relative_trans_i = relative_trans[i]
        #     relative_trans_j = relative_trans[i+1]

        #     # Compute the ground-truth relative translation vector
        #     relative_trans_ij = torch.matmul(relative_mat_i.transpose(0,1), relative_trans_i) - torch.matmul(relative_mat_j.transpose(0,1), relative_trans_j)
        #     if torch.norm(relative_trans_ij, 2) <= 0:
        #         continue
        #     relative_trans_ij_norm = relative_trans_ij / torch.norm(relative_trans_ij, 2)

        #     # Compute the predicted relative translation vector
        #     pred_trans_i = pred_trans[i]
        #     pred_trans_j = pred_trans[i+1]
        #     pred_trans_ij = torch.mm(relative_mat_i.transpose(0,1), pred_trans_i) - torch.mm(relative_mat_j.transpose(0,1), pred_trans_j)
        #     if torch.norm(pred_trans_ij, 2) <= 0:
        #         continue
        #     pred_trans_ij_norm = pred_trans_ij / torch.norm(pred_trans_ij, 2)

        #     # Compute the loss
        #     loss = torch.norm(relative_trans_ij_norm - pred_trans_ij_norm)