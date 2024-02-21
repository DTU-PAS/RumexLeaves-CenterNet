import argparse
import numpy as np
import os
import random
import warnings
import torch
import shutil
import math
import re
from rumexleaves_centernet.config import get_exp
import pytest


class TestBatchTraining:
    # only reproducable loss values with DC disabled! Hence, DC are not covered in these tests.

    def exec_train(self, exp_ident):
        exp_file = f"tests/test_integration/exp_files/{exp_ident}.py"
        args = argparse.Namespace()
        args.exp_file = exp_file
        args.resume = False
        args.pretrained_weights = None
        args.logger = "tensorboard"
        exp = get_exp(exp_file)
        args.experiment_name = exp.exp_name

        if exp.seed is not None:
            np.random.seed(exp.seed)
            random.seed(exp.seed)
            torch.manual_seed(exp.seed)
            torch.cuda.manual_seed(exp.seed)
            torch.cuda.manual_seed_all(exp.seed)  # multi-GPU
            torch.backends.cudnn.deterministic = True
            os.environ["PYTHONHASHSEED"] = str(exp.seed)
            warnings.warn(
                "You have chosen to seed training. This will turn on the CUDNN deterministic setting, "
                "which can slow down your training considerably! You may see unexpected behavior "
                "when restarting from checkpoints."
            )

        trainer = exp.get_trainer(args)
        trainer.train()

        with open(f"log/train/{exp_ident}/train_log.txt", "r") as f:
            log = f.read()
        lines = log.split("\n")
        for line in lines:
            if f"epoch: {exp.num_iterations + 1}/{exp.num_iterations + 1}" in line:
                losses = {
                    "loss": math.inf,
                    "hm_loss": math.inf,
                    "wh_loss": math.inf,
                    "off_loss": math.inf,
                    "kp_loss": math.inf,
                    "angle_loss": math.inf,
                    "kp_hm_loss": math.inf,
                }
                for k in losses.keys():
                    match = re.search(f" {k}: ([0-9]*.[0-9]*)", line)
                    if match:
                        losses[k] = round(float(match.group(1)), 2)

                break
        shutil.rmtree(f"log/train/{exp_ident}")
        return losses

    def test_train_1_batch_sincos_tlbr_segment_3(self):
        exp_ident = "train_1_batch_sincos_tlbr_segment_3"

        losses = self.exec_train(exp_ident)

        assert losses["loss"] == pytest.approx(18.80, 0.01)
        assert losses["hm_loss"] == pytest.approx(2.61, 0.01)
        assert losses["wh_loss"] == pytest.approx(0.21, 0.01)
        assert losses["off_loss"] == pytest.approx(0.46, 0.01)
        assert losses["kp_loss"] == pytest.approx(0.33, 0.01)
        assert losses["angle_loss"] == pytest.approx(0.0, 0.01)
        assert losses["kp_hm_loss"] == pytest.approx(10.33, 0.01)

    def test_train_1_batch_sincos_tlbr_segment_0(self):
        exp_ident = "train_1_batch_sincos_tlbr_segment_0"

        losses = self.exec_train(exp_ident)

        print(losses)

        assert losses["loss"] == pytest.approx(18.82, 0.01)
        assert losses["hm_loss"] == pytest.approx(2.58, 0.01)
        assert losses["wh_loss"] == pytest.approx(0.21, 0.01)
        assert losses["off_loss"] == pytest.approx(0.45, 0.01)
        assert losses["kp_loss"] == pytest.approx(0.35, 0.01)
        assert losses["angle_loss"] == pytest.approx(0.0, 0.01)
        assert losses["kp_hm_loss"] == pytest.approx(10.18, 0.01)

    def test_train_1_batch_sincos_tlbr_segment_7(self):
        exp_ident = "train_1_batch_sincos_tlbr_segment_7"

        losses = self.exec_train(exp_ident)

        assert losses["loss"] == pytest.approx(18.84, 0.01)
        assert losses["hm_loss"] == pytest.approx(2.63, 0.01)
        assert losses["wh_loss"] == pytest.approx(0.21, 0.01)
        assert losses["off_loss"] == pytest.approx(0.41, 0.01)
        assert losses["kp_loss"] == pytest.approx(0.33, 0.01)
        assert losses["angle_loss"] == pytest.approx(0.0, 0.01)
        assert losses["kp_hm_loss"] == pytest.approx(10.44, 0.01)

    def test_train_1_batch_sincos_tlbr_point_3(self):
        exp_ident = "train_1_batch_sincos_tlbr_point_3"

        losses = self.exec_train(exp_ident)

        assert losses["loss"] == pytest.approx(22.05, 0.01)
        assert losses["hm_loss"] == pytest.approx(2.57, 0.01)
        assert losses["wh_loss"] == pytest.approx(0.21, 0.01)
        assert losses["off_loss"] == pytest.approx(0.46, 0.01)
        assert losses["kp_loss"] == pytest.approx(0.33, 0.01)
        assert losses["angle_loss"] == pytest.approx(0.0, 0.01)
        assert losses["kp_hm_loss"] == pytest.approx(13.63, 0.01)

    def test_train_1_batch_360_tlbr_segment_3(self):
        exp_ident = "train_1_batch_360_tlbr_segment_3"

        losses = self.exec_train(exp_ident)

        assert losses["loss"] == pytest.approx(20.26, 0.01)
        assert losses["hm_loss"] == pytest.approx(2.53, 0.01)
        assert losses["wh_loss"] == pytest.approx(0.14, 0.01)
        assert losses["off_loss"] == pytest.approx(0.46, 0.01)
        assert losses["kp_loss"] == pytest.approx(0.23, 0.01)
        assert losses["angle_loss"] == pytest.approx(0.0, 0.01)
        assert losses["kp_hm_loss"] == pytest.approx(13.62, 0.01)

    def test_train_1_batch_sincos_wh_segment(self):
        exp_ident = "train_1_batch_sincos_wh_segment"

        losses = self.exec_train(exp_ident)

        print(losses)

        assert losses["loss"] == pytest.approx(19.44, 0.01)
        assert losses["hm_loss"] == pytest.approx(2.55, 0.01)
        assert losses["wh_loss"] == pytest.approx(0.32, 0.01)
        assert losses["off_loss"] == pytest.approx(0.51, 0.01)
        assert losses["kp_loss"] == pytest.approx(0.28, 0.01)
        assert losses["angle_loss"] == pytest.approx(0.0, 0.01)
        assert losses["kp_hm_loss"] == pytest.approx(10.4, 0.01)
