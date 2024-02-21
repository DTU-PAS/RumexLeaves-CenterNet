import argparse
import random
import warnings

import torch
import torch.backends.cudnn as cudnn
from loguru import logger

from rumexleaves_centernet.config import Exp, get_exp
from rumexleaves_centernet.utils import load_ckpt


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
    parser.add_argument("--logger", default="tensorboard", type=str, help="tensorboard | wandb")
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


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

    # Model and loaded weights
    model = exp.get_model()
    device = exp.get_device()
    if args.weights is not None:
        ckpt = torch.load(args.weights, map_location=device)["model"]
    else:
        ckpt = torch.load(exp.ckpt, map_location=device)["model"]
    model = load_ckpt(model, ckpt)
    model.to(device)

    evaluator = exp.get_evaluator(args)
    (performance, loss_meter), bb_detections, kp_detections = exp.eval(model, evaluator)

    print(performance)


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file)
    exp.merge(args.opts)

    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    main(exp, args)
