import math

import cv2
import numpy as np


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.0) / 2.0 for ss in shape]
    y, x = np.ogrid[-m : m + 1, -n : n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top : y + bottom, x - left : x + right]
    masked_gaussian = gaussian[radius - top : radius + bottom, radius - left : radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def get_line_pixels_bresenham(x1, y1, x2, y2, width, height):
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    x, y = x1, y1
    sx = -1 if x1 > x2 else 1
    sy = -1 if y1 > y2 else 1
    pixels = []

    if dx > dy:
        err = dx / 2.0
        while x != x2:
            if 0 <= y < height and 0 <= x < width:
                pixels.append([x, y])
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy / 2.0
        while y != y2:
            if 0 <= y < height and 0 <= x < width:
                pixels.append([x, y])
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy
    if 0 <= y < height and 0 <= x < width:
        pixels.append([x, y])

    return pixels


def get_line_pixels(x1, y1, x2, y2, width, height):
    # Define a line using the endpoints (x1, y1) and (x2, y2)
    line_mask = np.zeros((width, height))
    cv2.line(line_mask, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), thickness=1)

    # Get the indices of the non-zero pixels in the line mask
    indices = np.transpose(np.array(np.where(line_mask != 0)))
    return indices


def draw_line_gaussian(heatmap, line, radius, k=1):
    pix = get_line_pixels(line[0], line[1], line[2], line[3], heatmap.shape[1], heatmap.shape[0])
    for p in pix:
        heatmap = draw_umich_gaussian(heatmap, p, radius, k)
    return heatmap


def gaussian_radius(det_size, min_overlap=0.7):
    height, width = det_size

    a1 = 1
    b1 = height + width
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1**2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2**2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3**2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    return min(r1, r2, r3)


def reformulate_target(image, num_classes, targets, target_conf, max_objects=80, down_ratio=4):
    w, h = image.shape[2], image.shape[1]
    output_h = h // down_ratio
    output_w = w // down_ratio

    # Init targets
    hm = np.zeros((num_classes, output_h, output_w), dtype=np.float32)
    off = np.zeros((max_objects, 2), dtype=np.float32)
    ind = np.zeros((max_objects), dtype=np.int64)
    reg_mask = np.zeros((max_objects), dtype=np.uint8)

    if target_conf["do_kp"]:
        mult = 2
        if target_conf["angle_mode"] == "sincos":
            mult = 3
        kp = np.zeros((max_objects, 8 * mult), dtype=np.float32)
        kp_reg_mask = np.zeros((max_objects, 8 * mult), dtype=np.float32)

    if target_conf["do_obb"]:
        dim = 3
        if target_conf["box_mode"] == "tlbr":
            dim += 2
        if target_conf["angle_mode"] == "sincos":
            dim += 1
        obb = np.zeros((max_objects, dim), dtype=np.float32)

    if target_conf["do_kphm"]:
        dim = 8
        if target_conf["kphm_mode"] == "segment":
            dim = 5
        kphm = np.zeros((dim, output_h, output_w), dtype=np.float32)

    cp_conf = target_conf["cp_i"]

    # Centerpoint will be generated based on polyline
    # For bounding box we will provide tlbr with respect to centerpoint
    for i, (kp_target, bb_target) in enumerate(
        zip(targets["kp"]["target"][:max_objects], targets["bb"]["target"][:max_objects])
    ):
        kp_target /= down_ratio
        bb_target[:4] /= down_ratio

        # centerpoint (hm)
        if cp_conf == -1:
            ct = np.array([bb_target[0], bb_target[1]], dtype=np.float32)
        else:
            ct = np.array([kp_target[cp_conf][0], kp_target[cp_conf][1]], dtype=np.float32)
        ct_int = ct.astype(np.int32)
        cls_id = int(bb_target[5])
        radius = gaussian_radius((math.ceil(bb_target[2]), math.ceil(bb_target[3])))
        radius = max(0, int(radius))
        draw_umich_gaussian(hm[cls_id], ct_int, radius)
        ind[i] = ct_int[1] * output_w + ct_int[0]

        # offset (off)
        off[i] = ct - ct_int

        # OBB
        if target_conf["do_obb"]:
            width = bb_target[2]
            height = bb_target[3]
            angle = bb_target[4]

            # OBB Angle
            if target_conf["angle_mode"] == "360":
                obb[i][-1] = angle
                if target_conf["normalize"]:
                    obb[i][-1] /= 360.0
            elif target_conf["angle_mode"] == "sincos":
                angle_cos = (np.cos(angle * np.pi / 180) + 1) / 2  # [-1, 1] -> [0, 1]
                angle_sin = (np.sin(angle * np.pi / 180) + 1) / 2  # [-1, 1] -> [0, 1]
                obb[i][-2] = angle_cos
                obb[i][-1] = angle_sin

            # OBB Box
            if target_conf["box_mode"] == "wh":
                obb[i][1] = width
                obb[i][0] = height
                if target_conf["normalize"]:
                    obb[i][1] /= output_w
                    obb[i][0] /= output_h
            elif target_conf["box_mode"] == "tlbr":
                rel_ct = np.array([-bb_target[0] + ct[0], -bb_target[1] + ct[1]], dtype=np.float32)
                c, s = np.cos(-angle * np.pi / 180), np.sin(-angle * np.pi / 180)
                R = np.array(((c, -s), (s, c)))
                rot_rel_ct = (R @ rel_ct).T

                obb[i][0] = abs(-height / 2 - rot_rel_ct[1])
                obb[i][1] = abs(-width / 2 - rot_rel_ct[0])
                obb[i][2] = abs(height / 2 - rot_rel_ct[1])
                obb[i][3] = abs(width / 2 - rot_rel_ct[0])
                if target_conf["normalize"]:
                    denorm = np.array([output_h, output_w, output_h, output_w])
                    obb[i][:4] /= denorm

        if targets["bb"]["vis"][i]:
            reg_mask[i] = 1

        # Keypoints
        # Keypoints Regression
        if target_conf["do_kp"]:
            dist = np.linalg.norm(kp_target[:, :2] - ct, axis=1)
            angle = np.arctan2(kp_target[:, 1] - ct[1], kp_target[:, 0] - ct[0]) * 180 / math.pi
            angle[angle < 0] += 360

            if target_conf["normalize"]:
                dist = dist / output_w

            if target_conf["angle_mode"] == "360":
                if target_conf["normalize"]:
                    angle /= 360.0
                kp_target = np.zeros((8, 2))
                kp_target[:, 0] = dist
                kp_target[:, 1] = angle

            elif target_conf["angle_mode"] == "sincos":
                angle_cos = (np.cos(angle * np.pi / 180.0) + 1) / 2  # [-1, 1] -> [0, 1]
                angle_sin = (np.sin(angle * np.pi / 180.0) + 1) / 2  # [-1, 1] -> [0, 1]
                kp_target = np.zeros((8, 3))
                kp_target[:, 0] = dist
                kp_target[:, 1] = angle_cos
                kp_target[:, 2] = angle_sin
            kp[i] = kp_target.reshape(-1)

            if reg_mask[i] == 1:
                kp_reg_mask_temp = np.hstack([targets["kp"]["vis"][i], targets["kp"]["vis"][i]])
                if target_conf["angle_mode"] == "sincos":
                    kp_reg_mask_temp = np.hstack([kp_reg_mask_temp, targets["kp"]["vis"][i]])
                kp_reg_mask[i] = kp_reg_mask_temp.reshape(-1)

            # Keypoints Heatmap
            if target_conf["do_kphm"]:
                if target_conf["kphm_mode"] == "point":
                    for j, pos in enumerate(kp_target):
                        draw_umich_gaussian(kphm[j], pos, radius)
                elif target_conf["kphm_mode"] == "segment":
                    remap_kp_ind = [0, 1, 1, 2, 3, 3, 3, 4]
                    kp_line = [1, 2, 4, 5, 6]
                    for j, pos in enumerate(kp_target):
                        # draw_umich_gaussian(kphm[j], pos, radius)
                        if j in kp_line:
                            draw_line_gaussian(
                                kphm[remap_kp_ind[j]],
                                [kp_target[j - 1, 1], kp_target[j - 1, 0], kp_target[j, 1], kp_target[j, 0]],
                                radius,
                            )
                            if j == 2 or j == 6:
                                draw_line_gaussian(
                                    kphm[remap_kp_ind[j]],
                                    [kp_target[j, 1], kp_target[j, 0], kp_target[j + 1, 1], kp_target[j + 1, 0]],
                                    radius,
                                )
                        else:
                            draw_umich_gaussian(kphm[remap_kp_ind[j]], pos, radius)

    ret = {
        "img": image,
        "hm": hm,
        "ind": ind,
        "off": off,
        "reg_mask": reg_mask,
    }
    if target_conf["do_kp"]:
        ret["kp"] = kp
        ret["kp_reg_mask"] = kp_reg_mask

    if target_conf["do_obb"]:
        ret["obb"] = obb

    if target_conf["do_kphm"]:
        ret["kphm"] = kphm

    return ret


def targets_to_hm_off_tlbra_kp_kphm2(image, num_classes, targets, cp_i, max_objects=80, down_ratio=4, norm=False):
    w, h = image.shape[2], image.shape[1]
    output_h = h // down_ratio
    output_w = w // down_ratio

    # Init targets
    hm = np.zeros((num_classes, output_h, output_w), dtype=np.float32)
    kphm = np.zeros((5, output_h, output_w), dtype=np.float32)
    tlbra = np.zeros((max_objects, 5), dtype=np.float32)
    off = np.zeros((max_objects, 2), dtype=np.float32)
    ind = np.zeros((max_objects), dtype=np.int64)
    reg_mask = np.zeros((max_objects), dtype=np.uint8)
    kp = np.zeros((max_objects, 16), dtype=np.float32)
    kp_reg_mask = np.zeros((max_objects, 16), dtype=np.float32)

    # Centerpoint will be generated based on polyline
    # For bounding box we will provide tlbr with respect to centerpoint
    for i, (target, bb_target) in enumerate(
        zip(targets["kp"]["target"][:max_objects], targets["bb"]["target"][:max_objects])
    ):
        target /= down_ratio
        bb_target[:4] /= down_ratio

        # centerpoint (hm)
        ct = np.array([target[cp_i][0], target[cp_i][1]], dtype=np.float32)
        ct_int = ct.astype(np.int32)
        cls_id = int(bb_target[5])
        radius = gaussian_radius((math.ceil(bb_target[2]), math.ceil(bb_target[3])))
        radius = max(0, int(radius))
        draw_umich_gaussian(hm[cls_id], ct_int, radius)
        ind[i] = ct_int[1] * output_w + ct_int[0]

        # offset (off)
        off[i] = ct - ct_int

        # oriented bounding box (tlbr and angle)
        width = bb_target[2]
        height = bb_target[3]
        rel_ct = np.array([-bb_target[0] + ct[0], -bb_target[1] + ct[1]], dtype=np.float32)
        angle = bb_target[4]
        c, s = np.cos(-angle * np.pi / 180), np.sin(-angle * np.pi / 180)
        R = np.array(((c, -s), (s, c)))
        rot_rel_ct = (R @ rel_ct).T

        tlbra[i][0] = abs(-height / 2 - rot_rel_ct[1])
        tlbra[i][1] = abs(-width / 2 - rot_rel_ct[0])
        tlbra[i][2] = abs(height / 2 - rot_rel_ct[1])
        tlbra[i][3] = abs(width / 2 - rot_rel_ct[0])
        tlbra[i][4] = angle
        if norm:
            denorm = np.array([output_h, output_w, output_h, output_w])
            tlbra[i][:4] /= denorm
            tlbra[i][0][4] /= 360.0

        if targets["bb"]["vis"][i]:
            reg_mask[i] = 1

        # keypoints as heatmap
        remap_kp_ind = [0, 1, 1, 2, 3, 3, 3, 4]
        kp_line = [1, 2, 4, 5, 6]
        for j, pos in enumerate(target):
            # draw_umich_gaussian(kphm[j], pos, radius)
            if j in kp_line:
                draw_line_gaussian(
                    kphm[remap_kp_ind[j]], [target[j - 1, 1], target[j - 1, 0], target[j, 1], target[j, 0]], radius
                )
                if j == 2 or j == 6:
                    draw_line_gaussian(
                        kphm[remap_kp_ind[j]], [target[j, 1], target[j, 0], target[j + 1, 1], target[j + 1, 0]], radius
                    )
            else:
                draw_umich_gaussian(kphm[remap_kp_ind[j]], pos, radius)

        # keypoints as distance and angle
        dist = np.linalg.norm(target[:, :2] - ct, axis=1)
        angle = np.arctan2(target[:, 1] - ct[1], target[:, 0] - ct[0]) * 180 / math.pi
        angle[angle < 0] += 360
        if norm:
            dist = dist / output_w
            angle = angle / 360.0
        target[:, 0] = dist
        target[:, 1] = angle
        kp[i] = target.reshape(-1)
        if reg_mask[i] == 1:
            kp_reg_mask[i] = np.hstack([targets["kp"]["vis"][i], targets["kp"]["vis"][i]]).reshape(-1)

    ret = {
        "img": image,
        "hm": hm,
        "reg_mask": reg_mask,
        "ind": ind,
        "wh": tlbra,
        "off": off,
        "kp": kp,
        "kp_reg_mask": kp_reg_mask,
        "kphm": kphm,
    }
    return ret


def targets_to_hm_off_tlbra_kp_kphm(image, num_classes, targets, cp_i, max_objects=80, down_ratio=4, norm=False):
    w, h = image.shape[2], image.shape[1]
    output_h = h // down_ratio
    output_w = w // down_ratio

    # Init targets
    hm = np.zeros((num_classes, output_h, output_w), dtype=np.float32)
    kphm = np.zeros((8, output_h, output_w), dtype=np.float32)
    tlbra = np.zeros((max_objects, 5), dtype=np.float32)
    off = np.zeros((max_objects, 2), dtype=np.float32)
    ind = np.zeros((max_objects), dtype=np.int64)
    reg_mask = np.zeros((max_objects), dtype=np.uint8)
    kp = np.zeros((max_objects, 16), dtype=np.float32)
    kp_reg_mask = np.zeros((max_objects, 16), dtype=np.float32)

    # Centerpoint will be generated based on polyline
    # For bounding box we will provide tlbr with respect to centerpoint
    for i, (target, bb_target) in enumerate(
        zip(targets["kp"]["target"][:max_objects], targets["bb"]["target"][:max_objects])
    ):
        target /= down_ratio
        bb_target[:4] /= down_ratio

        # centerpoint (hm)
        ct = np.array([target[cp_i][0], target[cp_i][1]], dtype=np.float32)
        ct_int = ct.astype(np.int32)
        cls_id = int(bb_target[5])
        radius = gaussian_radius((math.ceil(bb_target[2]), math.ceil(bb_target[3])))
        radius = max(0, int(radius))
        draw_umich_gaussian(hm[cls_id], ct_int, radius)
        ind[i] = ct_int[1] * output_w + ct_int[0]

        # offset (off)
        off[i] = ct - ct_int

        # oriented bounding box (tlbr and angle)
        width = bb_target[2]
        height = bb_target[3]
        rel_ct = np.array([-bb_target[0] + ct[0], -bb_target[1] + ct[1]], dtype=np.float32)
        angle = bb_target[4]
        c, s = np.cos(-angle * np.pi / 180), np.sin(-angle * np.pi / 180)
        R = np.array(((c, -s), (s, c)))
        rot_rel_ct = (R @ rel_ct).T

        tlbra[i][0] = abs(-height / 2 - rot_rel_ct[1])
        tlbra[i][1] = abs(-width / 2 - rot_rel_ct[0])
        tlbra[i][2] = abs(height / 2 - rot_rel_ct[1])
        tlbra[i][3] = abs(width / 2 - rot_rel_ct[0])
        tlbra[i][0] = angle
        if norm:
            denorm = np.array([output_h, output_w, output_h, output_w])
            tlbra[i][:4] /= denorm
            tlbra[i][0][4] /= 360.0

        if targets["bb"]["vis"][i]:
            reg_mask[i] = 1

        # keypoints as heatmap
        for j, pos in enumerate(target):
            draw_umich_gaussian(kphm[j], pos, radius)

        # keypoints as distance and angle
        dist = np.linalg.norm(target[:, :2] - ct, axis=1)
        angle = np.arctan2(target[:, 1] - ct[1], target[:, 0] - ct[0]) * 180 / math.pi
        angle[angle < 0] += 360
        if norm:
            dist = dist / output_w
            angle = angle / 360.0
        target[:, 0] = dist
        target[:, 1] = angle
        kp[i] = target.reshape(-1)
        if reg_mask[i] == 1:
            kp_reg_mask[i] = np.hstack([targets["kp"]["vis"][i], targets["kp"]["vis"][i]]).reshape(-1)

    ret = {
        "img": image,
        "hm": hm,
        "reg_mask": reg_mask,
        "ind": ind,
        "wh": tlbra,
        "off": off,
        "kp": kp,
        "kp_reg_mask": kp_reg_mask,
        "kphm": kphm,
    }
    return ret


def targets_to_hm_off_tlbra_kp_sincos(image, num_classes, targets, cp_i, max_objects=80, down_ratio=4, norm=False):
    w, h = image.shape[2], image.shape[1]
    output_h = h // down_ratio
    output_w = w // down_ratio

    # Init targets
    hm = np.zeros((num_classes, output_h, output_w), dtype=np.float32)
    off = np.zeros((max_objects, 2), dtype=np.float32)
    tlbra = np.zeros((max_objects, 5), dtype=np.float32)
    ind = np.zeros((max_objects), dtype=np.int64)
    reg_mask = np.zeros((max_objects), dtype=np.uint8)
    kp = np.zeros((max_objects, 24), dtype=np.float32)
    kp_reg_mask = np.zeros((max_objects, 24), dtype=np.float32)

    # Centerpoint will be generated based on polyline
    # For bounding box we will provide tlbr with respect to centerpoint
    for i, (target, bb_target) in enumerate(
        zip(targets["kp"]["target"][:max_objects], targets["bb"]["target"][:max_objects])
    ):
        target /= down_ratio
        bb_target[:4] /= down_ratio

        # centerpoint (hm)
        ct = np.array([target[cp_i][0], target[cp_i][1]], dtype=np.float32)
        ct_int = ct.astype(np.int32)
        cls_id = int(bb_target[5])
        radius = gaussian_radius((math.ceil(bb_target[2]), math.ceil(bb_target[3])))
        radius = max(0, int(radius))
        draw_umich_gaussian(hm[cls_id], ct_int, radius)
        ind[i] = ct_int[1] * output_w + ct_int[0]

        # offset (off)
        off[i] = ct - ct_int

        # oriented bounding box (tlbr and angle)
        width = bb_target[2]
        height = bb_target[3]
        rel_ct = np.array([-bb_target[0] + ct[0], -bb_target[1] + ct[1]], dtype=np.float32)
        angle = bb_target[4]
        c, s = np.cos(-angle * np.pi / 180), np.sin(-angle * np.pi / 180)
        R = np.array(((c, -s), (s, c)))
        rot_rel_ct = (R @ rel_ct).T

        tlbra[i][0] = abs(-height / 2 - rot_rel_ct[1])
        tlbra[i][1] = abs(-width / 2 - rot_rel_ct[0])
        tlbra[i][2] = abs(height / 2 - rot_rel_ct[1])
        tlbra[i][3] = abs(width / 2 - rot_rel_ct[0])
        tlbra[i][4] = angle
        if norm:
            denorm = np.array([output_h, output_w, output_h, output_w])
            tlbra[i][:4] /= denorm
            tlbra[i][4] /= 360.0

        if targets["bb"]["vis"][i]:
            reg_mask[i] = 1

        # keypoints as distance and angle
        dist = np.linalg.norm(target[:, :2] - ct, axis=1)
        angle = np.arctan2(target[:, 1] - ct[1], target[:, 0] - ct[0])

        angle_cos = (np.cos(angle) + 1) / 2  # [-1, 1] -> [0, 1]
        angle_sin = (np.sin(angle) + 1) / 2  # [-1, 1] -> [0, 1]
        if norm:
            dist = dist / output_w
        kp[i, ::3] = dist
        kp[i, 1::3] = angle_cos
        kp[i, 2::3] = angle_sin
        if reg_mask[i] == 1:
            kp_reg_mask[i] = np.hstack(
                [targets["kp"]["vis"][i], targets["kp"]["vis"][i], targets["kp"]["vis"][i]]
            ).reshape(-1)

    ret = {
        "img": image,
        "hm": hm,
        "reg_mask": reg_mask,
        "ind": ind,
        "off": off,
        "wh": tlbra,
        "kp": kp,
        "kp_reg_mask": kp_reg_mask,
    }
    return ret


def targets_to_hm_off_tlbra_kp(image, num_classes, targets, cp_i, max_objects=80, down_ratio=4, norm=False):
    w, h = image.shape[2], image.shape[1]
    output_h = h // down_ratio
    output_w = w // down_ratio

    # Init targets
    hm = np.zeros((num_classes, output_h, output_w), dtype=np.float32)
    tlbra = np.zeros((max_objects, 5), dtype=np.float32)
    off = np.zeros((max_objects, 2), dtype=np.float32)
    ind = np.zeros((max_objects), dtype=np.int64)
    reg_mask = np.zeros((max_objects), dtype=np.uint8)
    kp = np.zeros((max_objects, 16), dtype=np.float32)
    kp_reg_mask = np.zeros((max_objects, 16), dtype=np.float32)

    # Centerpoint will be generated based on polyline
    # For bounding box we will provide tlbr with respect to centerpoint
    for i, (target, bb_target) in enumerate(
        zip(targets["kp"]["target"][:max_objects], targets["bb"]["target"][:max_objects])
    ):
        target /= down_ratio
        bb_target[:4] /= down_ratio

        # centerpoint (hm)
        ct = np.array([target[cp_i][0], target[cp_i][1]], dtype=np.float32)
        ct_int = ct.astype(np.int32)
        cls_id = int(bb_target[5])
        radius = gaussian_radius((math.ceil(bb_target[2]), math.ceil(bb_target[3])))
        radius = max(0, int(radius))
        draw_umich_gaussian(hm[cls_id], ct_int, radius)
        ind[i] = ct_int[1] * output_w + ct_int[0]

        # offset (off)
        off[i] = ct - ct_int

        # oriented bounding box (tlbr and angle)
        width = bb_target[2]
        height = bb_target[3]
        rel_ct = np.array([-bb_target[0] + ct[0], -bb_target[1] + ct[1]], dtype=np.float32)
        angle = bb_target[4]
        c, s = np.cos(-angle * np.pi / 180), np.sin(-angle * np.pi / 180)
        R = np.array(((c, -s), (s, c)))
        rot_rel_ct = (R @ rel_ct).T

        tlbra[i][0] = abs(-height / 2 - rot_rel_ct[1])
        tlbra[i][1] = abs(-width / 2 - rot_rel_ct[0])
        tlbra[i][2] = abs(height / 2 - rot_rel_ct[1])
        tlbra[i][3] = abs(width / 2 - rot_rel_ct[0])
        tlbra[i][0] = angle
        if norm:
            denorm = np.array([output_h, output_w, output_h, output_w])
            tlbra[i][:4] /= denorm
            tlbra[i][4] /= 360.0

        if targets["bb"]["vis"][i]:
            reg_mask[i] = 1

        # keypoints as distance and angle
        dist = np.linalg.norm(target[:, :2] - ct, axis=1)
        angle = np.arctan2(target[:, 1] - ct[1], target[:, 0] - ct[0]) * 180 / math.pi
        angle[angle < 0] += 360
        if norm:
            dist = dist / output_w
            angle = angle / 360.0
        target[:, 0] = dist
        target[:, 1] = angle
        kp[i] = target.reshape(-1)
        if reg_mask[i] == 1:
            kp_reg_mask[i] = np.hstack([targets["kp"]["vis"][i], targets["kp"]["vis"][i]]).reshape(-1)

    ret = {
        "img": image,
        "hm": hm,
        "reg_mask": reg_mask,
        "ind": ind,
        "wh": tlbra,
        "off": off,
        "kp": kp,
        "kp_reg_mask": kp_reg_mask,
    }
    return ret


def targets_to_hm_off_wh_kp(image, num_classes, targets, max_objects=80, down_ratio=4, norm=False):
    w, h = image.shape[2], image.shape[1]
    output_h = h // down_ratio
    output_w = w // down_ratio

    # Init targets
    hm = np.zeros((num_classes, output_h, output_w), dtype=np.float32)
    wha = np.zeros((max_objects, 3), dtype=np.float32)
    off = np.zeros((max_objects, 2), dtype=np.float32)
    ind = np.zeros((max_objects), dtype=np.int64)
    reg_mask = np.zeros((max_objects), dtype=np.uint8)

    do_kp = len(targets["kp"]["target"]) > 0
    if do_kp:
        kp = np.zeros((max_objects, 16), dtype=np.float32)
        kp_reg_mask = np.zeros((max_objects, 16), dtype=np.float32)

    # Centerpoint will be generated based on bb center
    # hm, wh, off will be generated based on bbs
    for i, target in enumerate(targets["bb"]["target"][:max_objects]):
        target[:4] /= down_ratio
        width = target[2]
        height = target[3]
        angle = target[4]

        # centerpoint (hm)
        cls_id = int(target[5])
        radius = gaussian_radius((math.ceil(width), math.ceil(height)))
        radius = max(0, int(radius))
        ct = np.array([target[0], target[1]], dtype=np.float32)
        ct_int = ct.astype(np.int32)
        draw_umich_gaussian(hm[cls_id], ct_int, radius)
        ind[i] = ct_int[1] * output_w + ct_int[0]

        # offset (off)
        off[i] = ct - ct_int

        # oriented bounding box (wha)
        if norm:
            wha[i] = 1.0 * width / output_w, 1.0 * height / output_h, 1.0 * angle / 360.0
        else:
            wha[i] = 1.0 * width, 1.0 * height, 1.0 * angle

        if targets["bb"]["vis"][i]:
            reg_mask[i] = 1

        # keypoints (kp) as distance and angle
        if do_kp:
            kpt_i = targets["kp"]["target"][i, :, :] / down_ratio
            dist = np.linalg.norm(kpt_i[:, :2] - ct, axis=1)
            angle = np.arctan2(kpt_i[:, 1] - ct[1], kpt_i[:, 0] - ct[0]) * 180 / math.pi
            angle[angle < 0] += 360
            if norm:
                dist = dist / output_w
                angle = angle / 360.0
            kpt_i[:, 0] = dist
            kpt_i[:, 1] = angle
            kp[i] = kpt_i.reshape(-1)
            if reg_mask[i] == 1:
                kp_reg_mask[i] = np.hstack([targets["kp"]["vis"][i], targets["kp"]["vis"][i]]).reshape(-1)

    ret = {
        "img": image,
        "hm": hm,
        "reg_mask": reg_mask,
        "ind": ind,
        "wh": wha,
        "off": off,
    }

    if do_kp:
        ret["kp"] = kp
        ret["kp_reg_mask"] = kp_reg_mask
    return ret


def targets_to_hm_off_wh_kp_sincos(image, num_classes, targets, max_objects=80, down_ratio=4, norm=False):
    w, h = image.shape[2], image.shape[1]
    output_h = h // down_ratio
    output_w = w // down_ratio

    # Init targets
    hm = np.zeros((num_classes, output_h, output_w), dtype=np.float32)
    wha = np.zeros((max_objects, 3), dtype=np.float32)
    off = np.zeros((max_objects, 2), dtype=np.float32)
    ind = np.zeros((max_objects), dtype=np.int64)
    reg_mask = np.zeros((max_objects), dtype=np.uint8)

    do_kp = len(targets["kp"]["target"]) > 0
    if do_kp:
        kp = np.zeros((max_objects, 24), dtype=np.float32)
        kp_reg_mask = np.zeros((max_objects, 24), dtype=np.float32)

    # Centerpoint will be generated based on bb center
    # hm, wh, off will be generated based on bbs
    for i, target in enumerate(targets["bb"]["target"][:max_objects]):
        target[:4] /= down_ratio
        width = target[2]
        height = target[3]
        angle = target[4]

        # centerpoint (hm)
        cls_id = int(target[5])
        radius = gaussian_radius((math.ceil(width), math.ceil(height)))
        radius = max(0, int(radius))
        ct = np.array([target[0], target[1]], dtype=np.float32)
        ct_int = ct.astype(np.int32)
        draw_umich_gaussian(hm[cls_id], ct_int, radius)
        ind[i] = ct_int[1] * output_w + ct_int[0]

        # offset (off)
        off[i] = ct - ct_int

        # oriented bounding box (wha)
        if norm:
            wha[i] = 1.0 * width / output_w, 1.0 * height / output_h, 1.0 * angle / 360.0
        else:
            wha[i] = 1.0 * width, 1.0 * height, 1.0 * angle

        if targets["bb"]["vis"][i]:
            reg_mask[i] = 1

        # keypoints (kp) as distance and angle
        if do_kp:
            kpt_i = targets["kp"]["target"][i, :, :] / down_ratio
            dist = np.linalg.norm(kpt_i[:, :2] - ct, axis=1)
            angle = np.arctan2(kpt_i[:, 1] - ct[1], kpt_i[:, 0] - ct[0])
            angle_cos = (np.cos(angle) + 1) / 2  # [-1, 1] -> [0, 1]
            angle_sin = (np.sin(angle) + 1) / 2  # [-1, 1] -> [0, 1]
            if norm:
                dist = dist / output_w
            kp[i, ::3] = dist
            kp[i, 1::3] = angle_cos
            kp[i, 2::3] = angle_sin
            if reg_mask[i] == 1:
                kp_reg_mask[i] = np.hstack(
                    [targets["kp"]["vis"][i], targets["kp"]["vis"][i], targets["kp"]["vis"][i]]
                ).reshape(-1)

    ret = {
        "img": image,
        "hm": hm,
        "reg_mask": reg_mask,
        "ind": ind,
        "wh": wha,
        "off": off,
    }

    if do_kp:
        ret["kp"] = kp
        ret["kp_reg_mask"] = kp_reg_mask
    return ret


def targets_to_hm_off_wh(image, num_classes, targets, max_objects=50, down_ratio=4, norm=False):
    w, h = image.shape[2], image.shape[1]
    output_h = h // down_ratio
    output_w = w // down_ratio

    hm = np.zeros((num_classes, output_h, output_w), dtype=np.float32)
    wh = np.zeros((max_objects, 2), dtype=np.float32)
    off = np.zeros((max_objects, 2), dtype=np.float32)
    ind = np.zeros((max_objects), dtype=np.int64)
    reg_mask = np.zeros((max_objects), dtype=np.uint8)

    for i, target in enumerate(targets["bb"]["target"][:max_objects]):
        target[:4] /= down_ratio
        width = target[2]
        height = target[3]
        cls_id = int(target[4])
        radius = gaussian_radius((math.ceil(width), math.ceil(height)))
        radius = max(0, int(radius))
        ct = np.array([(target[0] + target[2] / 2), (target[1] + target[3] / 2)], dtype=np.float32)
        ct_int = ct.astype(np.int32)
        draw_umich_gaussian(hm[cls_id], ct_int, radius)
        if norm:
            wh[i] = 1.0 * width / output_w, 1.0 * height / output_h
        else:
            wh[i] = 1.0 * width, 1.0 * height

        ind[i] = ct_int[1] * output_w + ct_int[0]
        off[i] = ct - ct_int
        if targets["bb"]["vis"][i]:
            reg_mask[i] = 1

    ret = {
        "img": image,
        "hm": hm,
        "reg_mask": reg_mask,
        "ind": ind,
        "wh": wh,
        "off": off,
    }
    return ret
