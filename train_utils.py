import torch
import numpy as np
from src.losses.pm_loss import PyPMLoss
from src.losses.relative_camera_loss import loss_relative_rot, loss_relative_trans
from src.utils.relative_refine import pseudo_refine

def batch_data_self(cfg, data, file_names, model_teacher=None, device="cuda", phase="train"):
    # if phase != "train":
    #     return batch_data_test_self(cfg, data, device=device)
    out_dict = {}
    # batch self training data
    net_cfg = cfg.model.pose_net
    out_res = net_cfg.output_res

    tensor_kwargs = {"dtype": torch.float32, "device": device}
    assert model_teacher is not None
    assert not model_teacher.training, "teacher model must be in eval mode!"
    batch = {}
    for d in data:
      file_names.append(d["file_name"])
    
    # the image, infomation data and data from detection
    batch["image_id"] = torch.stack([torch.from_numpy(np.array(d["image_id"])) for d in data], dim=0).to(device, non_blocking=True)
    # augmented roi_image
    batch["roi_img"] = torch.stack([d["roi_img"] for d in data], dim=0).to(device, non_blocking=True)
    # original roi_image
    batch["roi_gt_img"] = torch.stack([d["roi_gt_img"] for d in data], dim=0).to(device, non_blocking=True)
    # original image
    batch["gt_img"] = torch.stack([d["gt_img"] for d in data], dim=0).to(device, non_blocking=True)
    im_H, im_W = batch["gt_img"].shape[-2:]

    batch["roi_cls"] = torch.tensor([d["roi_cls"] for d in data], dtype=torch.long).to(device, non_blocking=True)
    bs = batch["roi_cls"].shape[0]
    if "roi_coord_2d" in data[0]:
        batch["roi_coord_2d"] = torch.stack([d["roi_coord_2d"] for d in data], dim=0).to(
            device=device, non_blocking=True
        )
    
    batch["roi_cam"] = torch.stack([d["cam"] for d in data], dim=0).to(device, non_blocking=True)
    batch["roi_center"] = torch.stack([d["bbox_center"] for d in data], dim=0).to(
        device=device, dtype=torch.float32, non_blocking=True
    )
    batch["roi_scale"] = torch.tensor([d["scale"] for d in data], device=device, dtype=torch.float32)
    # for crop and resize
    rois_xy0 = batch["roi_center"] - batch["roi_scale"].view(bs, -1) / 2  # bx2
    rois_xy1 = batch["roi_center"] + batch["roi_scale"].view(bs, -1) / 2  # bx2
    batch["inst_rois"] = torch.cat([torch.arange(bs, **tensor_kwargs).view(-1, 1), rois_xy0, rois_xy1], dim=1)

    batch["roi_extent"] = torch.stack([d["roi_extent"] for d in data], dim=0).to(
        device=device, dtype=torch.float32, non_blocking=True
    )
    if "sym_info" in data[0]:
        batch["sym_info"] = [d["sym_info"] for d in data]

    if "roi_points" in data[0]:
        batch["roi_points"] = torch.stack([d["roi_points"] for d in data], dim=0).to(
            device=device, dtype=torch.float32, non_blocking=True
        )
    if "roi_fps_points" in data[0]:
        batch["roi_fps_points"] = torch.stack([d["roi_fps_points"] for d in data], dim=0).to(
            device=device, dtype=torch.float32, non_blocking=True
        )

    if cfg.model.pseudo_pose_type == "pose_refine" and "pose_refine" in data[0]:
        batch["pseudo_rot"] = torch.stack([d["pose_refine"][:3, :3] for d in data], dim=0).to(
            device=device, dtype=torch.float32, non_blocking=True
        )
        batch["pseudo_trans"] = torch.stack([d["pose_refine"][:3, 3] for d in data], dim=0).to(
            device=device, dtype=torch.float32, non_blocking=True
        )
    batch["roi_wh"] = torch.stack([d["roi_wh"] for d in data], dim=0).to(device, non_blocking=True)
    batch["resize_ratio"] = torch.tensor([d["resize_ratio"] for d in data]).to(
        device=device, dtype=torch.float32, non_blocking=True
    )
    batch["relative_rot"] = torch.stack([d["relative_rot"] for d in data], dim=0).to(device, non_blocking=True)
    batch["relative_trans"] = torch.stack([d["relative_trans"] for d in data], dim=0).to(device, non_blocking=True)
    # batch["pseudo_coor_x"] = coor_x = out_dict["coor_x"]
    # batch["pseudo_coor_y"] = coor_y = out_dict["coor_y"]
    # batch["pseudo_coor_z"] = coor_z = out_dict["coor_z"]

    # batch["pseudo_region"] = out_dict["region"]
    return batch

def compute_self_loss(
        cfg,
        batch,
        pred_rot,
        pred_trans,
):
    loss_config = cfg.train.self_loss
    loss_dict = {}

    # self pm loss
    if loss_config.pm_loss_cfg.loss_weight > 0:
        pm_loss_func = PyPMLoss(**loss_config.pm_loss_cfg)
        pseudo_refine(batch, batch["relative_rot"], batch["relative_trans"])
        loss_pm_dict = pm_loss_func(
            pred_rots=pred_rot,
            gt_rots=batch['pseudo_rot'],
            points=batch["roi_points"],
            pred_transes=pred_trans,
            gt_transes=batch["pseudo_trans"],
            extents=batch["roi_extent"],
            sym_infos=batch["sym_info"],
        )
        loss_dict.update(loss_pm_dict)
    
    # relative rot loss
    if loss_config.relative_rot_loss_cfg.loss_weight > 0:
        relative_rot_loss_func = loss_relative_rot(loss_config.relative_rot_loss_cfg)
        relative_rot_loss_dict = relative_rot_loss_func(
            relative_rot_quats=batch["relative_rot"],
            pred_rots=pred_rot,
            sym_infos=batch["sym_info"],
        )
        loss_dict.update(relative_rot_loss_dict)

    # relative trans loss
    if loss_config.relative_trans_loss_cfg.loss_weight > 0:
        relative_trans_loss_func = loss_relative_trans(loss_config.relative_trans_loss_cfg)
        relative_trans_loss_dict = relative_trans_loss_func(
            relative_trans = batch["relative_trans"],
            relative_rot_quats=batch["relative_rot"],
            pred_trans=pred_trans,
        )
        loss_dict.update(relative_trans_loss_dict)
    
    return loss_dict