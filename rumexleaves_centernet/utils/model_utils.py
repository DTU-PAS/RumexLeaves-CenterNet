#!/usr/bin/env python3
# Code partially taken from: https://github.com/Megvii-BaseDetection/YOLOX/blob/main/yolox/utils/model_utils.py
# -*- coding:utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import os
from copy import deepcopy
from typing import Sequence

import torch
import torch.nn as nn
from loguru import logger


def get_model_info(model: nn.Module, tsize: Sequence[int]) -> str:
    from thop import profile

    stride = 64
    img = torch.zeros((1, 3, stride, stride), device=next(model.parameters()).device)
    flops, params = profile(deepcopy(model), inputs=(img,), verbose=False)
    params /= 1e6
    flops /= 1e9
    flops *= tsize[0] * tsize[1] / stride / stride * 2  # Gflops
    info = "Params: {:.2f}M, Gflops: {:.2f}".format(params, flops)
    return info


def load_ckpt(model, ckpt, skip_heads=[]):
    model_state_dict = model.state_dict()
    load_dict = {}
    for key_model, v in model_state_dict.items():
        # Option for skipping network heads
        for head in skip_heads:
            if head in key_model:
                continue
        if key_model not in ckpt:
            logger.warning(
                "{} is not in the ckpt. \
                 Please double check and see if this is desired.".format(key_model)
            )
            continue
        v_ckpt = ckpt[key_model]
        if v.shape != v_ckpt.shape:
            logger.warning(
                "Shape of {} \
                 in checkpoint is {}, \
                 while shape of {} in model is {}.".format(key_model, v_ckpt.shape, key_model, v.shape)
            )
            continue
        load_dict[key_model] = v_ckpt

    model.load_state_dict(load_dict, strict=False)
    return model


def save_checkpoint(state, save_dir, ckpt_name=""):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    filename = os.path.join(save_dir, ckpt_name + "_ckpt.pth")
    torch.save(state, filename)


def _gather_feat(feat, ind, mask=None):
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat


def _sigmoid(x):
    y = torch.clamp(x.sigmoid_(), min=1e-4, max=1 - 1e-4)
    return y
