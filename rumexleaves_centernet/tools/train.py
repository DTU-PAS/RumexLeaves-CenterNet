import argparse
import os
import random
import warnings

import numpy as np
import torch
from loguru import logger

from rumexleaves_centernet.config import Exp, get_exp


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
    parser.add_argument("--resume", default=False, action="store_true", help="resume training")
    parser.add_argument("-ckpt", "--pretrained_weights", default=None, type=str, help="checkpoint file")
    parser.add_argument("--logger", default="wandb", type=str, help="tensorboard | wandb")
    parser.add_argument(
        "-e",
        "--start_epoch",
        default=None,
        type=int,
        help="resume training start epoch",
    )
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
        np.random.seed(exp.seed)
        random.seed(exp.seed)
        torch.manual_seed(exp.seed)
        torch.cuda.manual_seed(exp.seed)
        torch.cuda.manual_seed_all(exp.seed)  # multi-GPU
        torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
        os.environ["PYTHONHASHSEED"] = str(exp.seed)
        warnings.warn(
            "You have chosen to seed training. This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! You may see unexpected behavior "
            "when restarting from checkpoints."
        )

    trainer = exp.get_trainer(args)
    trainer.train()


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file)
    exp.merge(args.opts)

    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    main(exp, args)
