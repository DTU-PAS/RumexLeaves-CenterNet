import argparse
import random
import warnings

import albumentations as A
import albumentations.pytorch as AP
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from loguru import logger

from rumexleaves_centernet.config import Exp, get_exp
from rumexleaves_centernet.utils import ImageDrawer, ctdet_decode, kp_decode, load_ckpt


def make_parser():
    parser = argparse.ArgumentParser("train parser")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="experiment description file",
    )
    parser.add_argument("-ckpt", "--weights", default=None, type=str, help="checkpoint file")

    parser.add_argument("-img", "--image", default=None, type=str, help="image file")

    parser.add_argument("--logger", default="tensorboard", type=str, help="tensorboard | wandb")
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


def denorm(exp, predictions, device):
    output_size = (exp.input_size[0] / exp.down_ratio, exp.input_size[0] / exp.down_ratio)
    if "obb" in predictions:
        denorm_values = [output_size[0], output_size[1]]
        if exp.target_mode_conf["box_mode"] == "tlbr":
            denorm_values.extend([output_size[0], output_size[1]])
        if exp.target_mode_conf["angle_mode"] == "360":
            denorm_values.append(360.0)
        elif exp.target_mode_conf["angle_mode"] == "sincos":
            denorm_values.extend([1.0, 1.0])
        denorm = torch.tensor(denorm_values).to(device)
        predictions["obb"] *= denorm.unsqueeze(1).unsqueeze(1).expand(predictions["obb"].shape)
    if "kp" in predictions:
        denorm_values_dist = torch.ones((8, 1)) * output_size[0]
        if exp.target_mode_conf["angle_mode"] == "360":
            denorm_values_angle = [torch.ones((8, 1)) * 360]
        elif exp.target_mode_conf["angle_mode"] == "sincos":
            denorm_values_angle = [torch.ones((8, 1)), torch.ones((8, 1))]
        mult = torch.cat([denorm_values_dist, *denorm_values_angle], dim=1).reshape(-1).to(device)
        predictions["kp"] *= mult.unsqueeze(1).unsqueeze(1)
    return predictions


@logger.catch
def main(exp: Exp, args):
    if exp.seed is not None:
        random.seed(exp.seed)
        torch.manual_seed(exp.seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed training. This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! You may see unexpected behavior "
            "when restarting from checkpoints."
        )

    exp.center_thresh = 0.5

    # Model and loaded weights
    model = exp.get_model()
    device = exp.get_device()
    ckpt = torch.load(args.weights, map_location=device)["model"]
    model = load_ckpt(model, ckpt)
    model.to(device)

    # Load image and bring to expected input format
    augmentation_list1 = []
    augmentation_list1.append(A.LongestMaxSize(max_size=exp.input_size[0], interpolation=1, p=1.0))
    augmentation_list1.append(
        A.PadIfNeeded(
            min_height=exp.input_size[0],
            min_width=exp.input_size[1],
            border_mode=0,
            position="center",
            value=None,
            mask_value=None,
            p=1.0,
        )
    )

    augmentation_list2 = []

    if exp.p_normalize > 0:
        augmentation_list2.append(A.Normalize(mean=exp.ds_mean, std=exp.ds_std, p=exp.p_normalize, max_pixel_value=1.0))
    augmentation_list2.append(AP.ToTensorV2())

    img = cv2.imread(args.image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = A.Compose(augmentation_list1)(image=img)["image"]
    img_normalized = A.Compose(augmentation_list2)(image=img)["image"]
    img_normalized = img_normalized.to(device).unsqueeze(0)

    # Make Prediction
    predictions, loss_stats = model(img_normalized, None)

    # Decode Preciction
    if exp.norm_target:
        predictions = denorm(exp, predictions, device)

    kps = []
    if "kp" in predictions.keys():
        if exp.target_mode_conf["do_kphm"]:
            kphm = predictions["kphm"]
        else:
            kphm = None

        kps = kp_decode(
            predictions["hm"], predictions["kp"], exp.target_mode_conf["angle_mode"], kphm=kphm, reg=predictions["off"]
        )
        kps[:, :, :, :2] *= exp.down_ratio

    dets = torch.empty((0, 0)).to(device)
    if "obb" in predictions.keys():
        dets = ctdet_decode(
            predictions["hm"],
            predictions["obb"],
            exp.target_mode_conf,
            reg=predictions["off"],
        )
        dets[:, :, :4] *= exp.down_ratio

    img_with_prediction = img.copy()
    image_drawer = ImageDrawer(exp)
    image_drawer.draw_predicted_bb(img_with_prediction, dets[0])
    image_drawer.draw_pred_kps(img_with_prediction, kps[0])

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(img)
    ax[1].imshow(img_with_prediction)
    plt.show()

    for obbox in dets[0]:
        score = obbox[5]
        if score > exp.center_thresh:
            print(obbox)

    for kp_detection in kps[0]:
        score = kp_detection[0][2]
        if score > exp.center_thresh:
            print(score)


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file)
    exp.merge(args.opts)

    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    main(exp, args)
