# Code partially taken from: https://github.com/xingyizhou/CenterNet/blob/master/src/lib/models/decode.py
# Modified and extended by Ronja Güldenring, 2023
import math

import numpy as np
import torch
import torch.nn as nn

from rumexleaves_centernet.utils.model_utils import _gather_feat, _transpose_and_gather_feat


def _topk(scores, K=40):
    batch, cat, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()

    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_clses = (topk_ind / K).int()
    topk_inds = _gather_feat(topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


def _topk_channel(scores, K=40):
    batch, cat, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()

    return topk_scores, topk_inds, topk_ys, topk_xs


def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep


def ctpt_decode(heat, reg=None, K=100):
    batch, cat, height, width = heat.size()

    # heat = torch.sigmoid(heat)
    # perform nms on heatmaps
    heat = _nms(heat)

    scores, inds, clses, ys, xs = _topk(heat, K=K)
    if reg is not None:
        reg = _transpose_and_gather_feat(reg, inds)
        reg = reg.view(batch, K, 2)
        xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
        ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
    else:
        xs = xs.view(batch, K, 1) + 0.5
        ys = ys.view(batch, K, 1) + 0.5
    return scores, inds, clses, ys, xs


def kp_decode(heat, kps, angle_mode, kphm=None, kp_hm_offset=None, reg=None, K=100, device="cuda"):
    batch, cat, height, width = heat.size()
    scores, inds, clses, ys, xs = ctpt_decode(heat, reg=reg, K=K)
    clses = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)

    if angle_mode == "360":
        v = 2
    elif angle_mode == "sincos":
        v = 3

    num_kps = int(kps.shape[1] // v)

    kps = _transpose_and_gather_feat(kps, inds)
    kps = kps.view(batch, K, num_kps, v)

    if angle_mode == "360":
        angle = kps[:, :, :, 1] * math.pi / 180
    elif angle_mode == "sincos":
        angle = torch.arctan2(((kps[:, :, :, 2] * 2) - 1), ((kps[:, :, :, 1] * 2) - 1))

    kp_new = torch.zeros((batch, K, num_kps, 2)).to(device)
    kp_new[:, :, :, 0] = xs + torch.cos(angle) * kps[:, :, :, 0]
    kp_new[:, :, :, 1] = ys + torch.sin(angle) * kps[:, :, :, 0]
    kps = kp_new
    if kphm is not None:
        kps = map_regression_to_heatmap(kphm, kps, batch, num_kps, width, device, K, kp_hm_offset)

    kps = torch.cat(
        [
            kps,
            scores.unsqueeze(2).repeat(1, 1, num_kps, 1),
            clses.unsqueeze(2).repeat(1, 1, num_kps, 1),
        ],
        dim=3,
    )
    return kps


def map_regression_to_heatmap(kphm, kps, batch, num_kps, width, device, K, kphm_offset):
    kphm = _nms(kphm)
    if kphm.shape[1] == 5:
        kp_hm_extend = torch.zeros((kphm.shape[0], 8, kphm.shape[2], kphm.shape[3])).to(device)
        indices_map = [0, 1, 1, 2, 3, 3, 3, 4]
        for i, ind in enumerate(indices_map):
            kp_hm_extend[:, i, :, :] = kphm[:, ind, :, :]
        kphm = kp_hm_extend
    thresh = 0.1
    kps = kps.view(batch, K, num_kps, 2).permute(0, 2, 1, 3).contiguous()  # b x J x K x 2
    reg_kps = kps.unsqueeze(3).expand(batch, num_kps, K, K, 2)
    hm_score, hm_inds, hm_ys, hm_xs = _topk_channel(kphm, K=K)  # b x J x K

    if kphm_offset is not None:
        hp_offset = _transpose_and_gather_feat(kphm_offset, hm_inds.view(batch, -1))
        hp_offset = hp_offset.view(batch, num_kps, K, 2)
        hm_xs = hm_xs + hp_offset[:, :, :, 0]
        hm_ys = hm_ys + hp_offset[:, :, :, 1]
    else:
        hm_xs = hm_xs + 0.5
        hm_ys = hm_ys + 0.5

    mask = (hm_score > thresh).float()
    hm_score = (1 - mask) * -1 + mask * hm_score
    hm_ys = (1 - mask) * (-10000) + mask * hm_ys
    hm_xs = (1 - mask) * (-10000) + mask * hm_xs
    hm_kps = torch.stack([hm_xs, hm_ys], dim=-1).unsqueeze(2).expand(batch, num_kps, K, K, 2)
    dist = ((reg_kps - hm_kps) ** 2).sum(dim=4) ** 0.5
    min_dist, min_ind = dist.min(dim=3)  # b x J x K
    hm_score = hm_score.gather(2, min_ind).unsqueeze(-1)  # b x J x K x 1
    min_dist = min_dist.unsqueeze(-1)
    min_ind = min_ind.view(batch, num_kps, K, 1, 1).expand(batch, num_kps, K, 1, 2)
    hm_kps = hm_kps.gather(3, min_ind)
    hm_kps = hm_kps.view(batch, num_kps, K, 2)
    mask = (hm_score < thresh) + (min_dist > (width * 0.1))
    mask = (mask > 0).float().expand(batch, num_kps, K, 2)
    kps = (1 - mask) * hm_kps + mask * kps
    kps = kps.permute(0, 2, 1, 3)
    return kps


def ctdet_decode(heat, obb, target_mode_conf, reg=None, K=100):
    batch, cat, height, width = heat.size()
    scores, inds, clses, ys, xs = ctpt_decode(heat, reg=reg, K=K)
    obb = _transpose_and_gather_feat(obb, inds)
    obb = obb.view(batch, K, obb.shape[-1])
    clses = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)

    if target_mode_conf["angle_mode"] == "360":
        if target_mode_conf["box_mode"] == "wh":
            angles = obb[..., 2].view(batch, K, 1) * torch.pi / 180
        elif target_mode_conf["box_mode"] == "tlbr":
            angles = obb[..., 4].view(batch, K, 1) * torch.pi / 180
    elif target_mode_conf["angle_mode"] == "sincos":
        if target_mode_conf["box_mode"] == "wh":
            angles = torch.arctan2(((obb[..., 3] * 2) - 1), ((obb[..., 2] * 2) - 1)).view(batch, K, 1)
        elif target_mode_conf["box_mode"] == "tlbr":
            angles = torch.arctan2(((obb[..., 5] * 2) - 1), ((obb[..., 4] * 2) - 1)).view(batch, K, 1)

    if target_mode_conf["box_mode"] == "wh":
        bboxes = torch.cat([xs, ys, obb[..., 1:2], obb[..., 0:1]], dim=2)
    elif target_mode_conf["box_mode"] == "tlbr":
        wh_bb = torch.cat([obb[..., 1:2] + obb[..., 3:4], obb[..., 0:1] + obb[..., 2:3]], dim=2)
        c, s = torch.cos(angles), torch.sin(angles)
        R = torch.stack([c, -s, s, c], dim=2).view(batch, K, 2, 2)
        c_rel = torch.cat([wh_bb[..., 0:1] / 2 - obb[..., 1:2], wh_bb[..., 1:2] / 2 - obb[..., 0:1]], dim=2)
        R = R.reshape(-1, R.shape[2], R.shape[3])
        c_rel = c_rel.reshape(-1, c_rel.shape[2]).unsqueeze(2)
        c_rel_rot = torch.bmm(R, c_rel)
        c_rel_rot = c_rel_rot.reshape(batch, K, c_rel_rot.shape[1])
        bb_x = xs + c_rel_rot[..., 0:1]
        bb_y = ys + c_rel_rot[..., 1:2]
        bboxes = torch.cat([bb_x, bb_y, wh_bb], dim=2)

    detections = torch.cat([bboxes, angles * 180 / math.pi, scores, clses], dim=2)

    return detections


def obbox_2_box_corners(cx, cy, tlbr, angle):
    # We need to get the center of the bounding box
    width = tlbr[1] + tlbr[3]
    height = tlbr[0] + tlbr[2]
    angle *= np.pi / 180
    c, s = np.cos(angle), np.sin(angle)
    R = np.array(((c, -s), (s, c)))
    c_rel = np.array([width / 2 - tlbr[1], height / 2 - tlbr[0]])
    c_rel_rot = (R @ c_rel).T
    bbx = cx + c_rel_rot[0]
    bby = cy + c_rel_rot[1]

    # Lets get the coordinates of the bounding box corners
    bb_corners = np.array(
        [
            [-width / 2, -height / 2],
            [width / 2, -height / 2],
            [width / 2, height / 2],
            [-width / 2, height / 2],
        ]
    )
    bb_corners = (R @ bb_corners.T).T
    bb_corners += np.array([bbx, bby])
    return bb_corners
