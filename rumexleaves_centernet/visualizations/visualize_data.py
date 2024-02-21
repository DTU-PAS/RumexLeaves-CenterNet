import math
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np

from rumexleaves_centernet.data import RumexLeavesDataset, TrainTransform, ValTransform
from rumexleaves_centernet.utils import img_tensor_to_numpy, obbox_2_box_corners


def draw_obbs(img, ret, target_mode_conf, img_w, down_ratio, colors):
    for k, (index, bb, offset, vis) in enumerate(zip(ret["ind"], ret["obb"], ret["off"], ret["reg_mask"])):
        if index == 0:
            break
        if vis == 0:
            y_i = (math.floor(index // (img_w // down_ratio))) * down_ratio
            x_i = (index % (img_w // down_ratio)) * down_ratio
            print(f"*continue: x: {x_i}, y: {y_i}")
            continue
        y_i = (math.floor(index // (img_w // down_ratio))) * down_ratio
        x_i = (index % (img_w // down_ratio)) * down_ratio

        if target_mode_conf["box_mode"] == "tlbr":
            bb[:4] *= down_ratio
            if target_mode_conf["angle_mode"] == "sincos":
                angle = np.arctan2(((bb[5] * 2) - 1), ((bb[4] * 2) - 1)) * 180 / math.pi
            else:
                angle = bb[4]
            bb_corners = obbox_2_box_corners(x_i, y_i, bb[:4], angle)

        if target_mode_conf["box_mode"] == "wh":
            height = bb[0] * down_ratio
            width = bb[1] * down_ratio
            if target_mode_conf["angle_mode"] == "sincos":
                angle = np.arctan2(((bb[3] * 2) - 1), ((bb[2] * 2) - 1)) * 180 / math.pi
            else:
                angle = bb[2] * math.pi / 180

            # angle = bb[2] * math.pi/180
            tlbr = np.array([height / 2, width / 2, height / 2, width / 2])
            bb_corners = obbox_2_box_corners(x_i, y_i, tlbr, angle)

        cv2.drawContours(img, [bb_corners.astype(int)], 0, colors[k], 2)

        cv2.circle(img, (int(x_i), int(y_i)), 13, colors[k], 2)
        cv2.putText(
            img,
            f"{math.floor(angle)}",
            (int(x_i), int(y_i)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            colors[k],
            2,
            cv2.LINE_AA,
        )
    return img


def draw_keypoints(img, ret, target_mode_conf, img_w, down_ratio, colors):
    for k, (index, kp, offset, kp_reg_mask, vis) in enumerate(
        zip(ret["ind"], ret["kp"], ret["off"], ret["kp_reg_mask"], ret["reg_mask"])
    ):
        if vis == 0:
            continue
        y_i = math.floor(index // (img_w // down_ratio))
        x_i = index % (img_w // down_ratio)
        kp = kp.reshape(8, -1)
        stem = np.zeros((8, 2))
        for i, kp_point in enumerate(kp):
            if target_mode_conf["angle_mode"] == "sincos":
                angle = np.arctan2(((kp_point[2] * 2) - 1), ((kp_point[1] * 2) - 1)) * 180 / math.pi
            else:
                angle = kp_point[1]
            stem[i, 0] = x_i + math.cos(angle * math.pi / 180) * kp_point[0]
            stem[i, 1] = y_i + math.sin(angle * math.pi / 180) * kp_point[0]
        stem = np.multiply(stem, down_ratio)
        kp_reg_mask = kp_reg_mask.reshape(-1, 2)
        kp_reg_mask = kp_reg_mask[kp_reg_mask[:, 0].squeeze() == 1, :]
        for stem_point in stem:
            cv2.circle(img, (int(stem_point[0]), int(stem_point[1])), 3, colors[k], 3)
        cv2.polylines(img, [stem.astype(int)], 0, colors[k], 2)
    return img


if __name__ == "__main__":
    data_folder = "data/processed/RumexLeaves/iNaturalist"
    splits = ["random_train.txt"]
    img_list = []
    transform = "train"
    image_size = (512, 512)
    target_mode_conf = {
        "do_kp": True,
        "do_kphm": True,
        "do_obb": True,
        "angle_mode": "sincos",  # 360, sincos
        "box_mode": "tlbr",  # wh, tlrb
        "kphm_mode": "segment",  # point, segment
        "cp_i": 3,  # -1 (center point of bounding box), 0, 3, 7
        "normalize": False,
    }

    for s in splits:
        with open(f"{data_folder}/dataset_splits/{s}", "r+") as f:
            img_list = [line.replace("\n", "") for line in f.readlines()]

    if transform == "train":
        transform = TrainTransform(
            50,
            image_size=image_size,
            p_flip=0.0,
            p_rotate=0.0,
            p_color_jitter=0.0,
            p_rgb_shift=0.0,
            rgb_shift_values=[30, 50, 30],
            jitter_values=[0.2, 0.2, 0.2, 0.2],
            p_blur=0.0,
            p_random_brightness_contrast=0.0,
            p_normalize=0.0,
            p_scale_shift=0.0,
        )
    else:
        transform = ValTransform(
            50,
            image_size=image_size,
            p_normalize=0.0,
        )

    dataset = RumexLeavesDataset(
        data_dir=data_folder,
        image_list=img_list,
        classes=["leaf_blade"],
        preproc=transform,
        target_mode_conf=target_mode_conf,
        annotation_file_rel_to_img="annotations_oriented_bb.xml",
    )

    print(f"Dataset length: {len(dataset)}")

    for j in range(0, 5, 1):
        j = random.randint(0, len(dataset) - 1)
        ret = dataset[j]
        img = img_tensor_to_numpy(ret["img"])
        img_raw = img.copy()

        img_h, img_w = img.shape[:2]
        down_ratio = 4

        # automatically generate RGB color codes for each occurence
        colors = [(np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)) for _ in range(50)]

        # Draw Oriented Bounding Boxes
        draw_obbs(img, ret, target_mode_conf, img_w, down_ratio, colors)

        # Draw Keypoints
        if "kp" in ret.keys():
            draw_keypoints(img, ret, target_mode_conf, img_w, down_ratio, colors)

        # Plotting
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(img_raw)
        ax[1].imshow(img)
        plt.show(block=True)
