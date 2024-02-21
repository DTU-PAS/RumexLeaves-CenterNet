#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
# Code originally from: https://github.com/Megvii-BaseDetection/YOLOX/blob/main/yolox/core/trainer.py
# Modified by Ronja Güldenring

import argparse
import math
import os
import shutil
import time

import optuna
import torch
from loguru import logger
from torch.utils.tensorboard import SummaryWriter

from rumexleaves_centernet.config import Exp
from rumexleaves_centernet.utils import (
    MeterBuffer,
    WandbLogger,
    get_model_info,
    gpu_mem_usage,
    load_ckpt,
    save_checkpoint,
    setup_logger,
)


class Trainer:
    """
    Core class to train a model.
    ...

    Attributes
    ----------
    exp : Exp
        Experiment configuration
    args :
        Additional arguments from the command line
    """

    def __init__(self, exp: Exp, args: argparse.Namespace, trial) -> None:
        """
        Sets up training configuration.
        """
        self.exp = exp
        self.args = args
        self.trial = trial

        # training related attr
        self.num_iterations = exp.num_iterations
        self.device = exp.get_device()

        # data/dataloader related attr
        self.input_size = exp.input_size
        self.best_aps = {
            "best_average_ap50": 0,
            "best_mAP50": 0,
            "best_OKS50": 0,
            "best_OKS95": 0,
        }

        # metric record
        self.meter = MeterBuffer(window_size=exp.print_interval)
        self.file_name = os.path.join(exp.output_dir, args.experiment_name)

        os.makedirs(self.file_name, exist_ok=True)
        if self.exp.debug:
            self.debug_folder = os.path.join(self.file_name, "debug")
            os.makedirs(self.debug_folder, exist_ok=True)

        setup_logger(
            self.file_name,
            filename="train_log.txt",
            mode="a",
        )

    def train(self) -> None:
        """
        train function

        Returns
        -------
        None
        """
        self.before_train()
        try:
            self.train_in_epoch()
        except Exception:
            raise
        finally:
            self.after_train()

    def before_train(self) -> None:
        """
        Initializes/logs parameters relevant for training, such as optimizer, model, data loader, lr scheduler etc.

        Returns
        -------
        None
        """
        logger.info("args: {}".format(self.args))
        logger.info("exp value:\n{}".format(self.exp))
        shutil.copy(self.args.exp_file, self.file_name)

        # model related init
        model = self.exp.get_model()
        logger.info("Model Summary: {}".format(get_model_info(model, self.exp.input_size)))
        model.to(self.device)

        # solver related init
        self.optimizer = self.exp.get_optimizer(model)

        # value of epoch will be set in `resume_train`
        self.model = self.resume_train(model)

        self.epoch = 0

        # data related init
        self.train_loader = self.exp.get_data_loader("train")

        # max_iter means iters per epoch
        self.max_iter = len(self.train_loader)
        self.max_epoch = math.ceil(self.num_iterations // self.max_iter + 1)

        self.lr_scheduler = self.exp.get_lr_scheduler(self.max_iter, self.max_epoch)

        self.evaluator = self.exp.get_evaluator(self.args)
        # Tensorboard and Wandb loggers
        if self.args.logger == "tensorboard":
            self.tblogger = SummaryWriter(os.path.join(self.file_name, "tensorboard"))
        elif self.args.logger == "wandb":
            self.wandb_logger = WandbLogger.initialize_wandb_logger(
                self.args, self.exp, self.evaluator.val_loader.dataset
            )
        else:
            raise ValueError("logger must be either 'tensorboard' or 'wandb'")

        logger.info("Training start...")
        logger.info("\n{}".format(model))

    def before_epoch(self) -> None:
        """
        Initializes/Logs parameters relevant for training an epoch, such as progress in epoch.

        Returns
        -------
        None
        """
        logger.info("---> start train epoch{}".format(self.epoch + 1))

    def train_in_epoch(self) -> None:
        """
        Trains the model for one epoch.

        Returns
        -------
        None
        """
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.before_epoch()
            self.train_in_iter()
            self.after_epoch()

    def after_epoch(self) -> None:
        """
        Wraps up the training of one epoch, such as saving checkpoints, and evaluating the model.

        Returns
        -------
        None
        """
        if (self.epoch + 1) % self.exp.eval_interval == 0:
            update_best_ckpt = self.evaluate_and_save_model()
            self.save_ckpt(ckpt_name="latest", update_best_ckpt=update_best_ckpt)

    def before_iter(self) -> None:
        """
        Initializes parameters relevant for iterating over the training data.

        Returns
        -------
        None
        """
        self.model.train()

    def train_in_iter(self) -> None:
        """
        Iterating over the training data.

        Returns
        -------
        None
        """
        for iter, batch in enumerate(self.train_loader):
            self.before_iter()
            self.train_one_iter(iter, batch)
            self.after_iter()

    def train_one_iter(self, iter: int, batch: dict) -> None:
        """
        Training the model for one iteration, i.e. forward, backward, update learning rate etc.

        Returns
        -------
        None
        """
        self.iter = iter
        total_iter = self.progress_in_iter
        iter_start_time = time.time()

        # data
        targets = {k: batch[k] for k in batch.keys() if k != "img" and k != "meta"}
        for k in targets:
            targets[k] = targets[k].to(self.device)
        imgs = batch["img"].to(self.device)
        data_end_time = time.time()

        # forward
        _, loss_stats = self.model(imgs, targets)
        loss = loss_stats["loss"]

        # backward
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # update learning rate
        lr = self.lr_scheduler.update_lr(total_iter)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        iter_end_time = time.time()

        # logging
        self.meter.update(
            iter_time=iter_end_time - iter_start_time,
            data_time=data_end_time - iter_start_time,
            lr=lr,
            **loss_stats,
        )

    def after_iter(self) -> None:
        """
        Wraps up the training of one iteration, such as logging stats.

        Returns
        -------
        None
        """
        if (self.iter + 1) % self.exp.print_interval == 0:
            progress_str = "epoch: {}/{}, iter: {}/{}".format(
                self.epoch + 1, self.max_epoch, self.iter + 1, self.max_iter
            )
            loss_meter = self.meter.get_filtered_meter("loss")
            loss_str = ", ".join(["{}: {:.3f}".format(k, v.latest) for k, v in loss_meter.items()])

            time_meter = self.meter.get_filtered_meter("time")
            time_str = ", ".join(["{}: {:.3f}s".format(k, v.avg) for k, v in time_meter.items()])

            logger.info(
                "{}, mem: {:.0f}Mb, {}, {}, lr: {:.3e}".format(
                    progress_str,
                    gpu_mem_usage(),
                    time_str,
                    loss_str,
                    self.meter["lr"].latest,
                )
                + (", size: ({:d},{:d})".format(self.input_size[0], self.input_size[1]))
            )

            if self.args.logger == "tensorboard":
                self.tblogger.add_scalar("lr", self.meter["lr"].latest, self.progress_in_iter)
                for k, v in loss_meter.items():
                    self.tblogger.add_scalar(f"train/{k}", v.latest, self.progress_in_iter)
            if self.args.logger == "wandb":
                metrics = {"train/" + k: v.latest for k, v in loss_meter.items()}
                metrics.update({"train/lr": self.meter["lr"].latest})
                self.wandb_logger.log_metrics(metrics, step=self.progress_in_iter)

            self.meter.clear_meters()

    def after_train(self) -> None:
        """
        Wraps up the training of the experiment, such as logging the best AP.

        Returns
        -------
        None
        """
        logger.info("Training of experiment is done and the best APs of:")
        for k, v in self.best_aps.items():
            logger.info(f"{k}: {v:.3f}")

        if self.args.logger == "wandb":
            self.wandb_logger.finish()

    @property
    def progress_in_iter(self) -> int:
        """Compute the progress in iteration."""
        return self.epoch * self.max_iter + self.iter

    def resume_train(self, model: torch.nn.Module) -> torch.nn.Module:
        """
        Resume the training from the latest checkpoint or a specfic checkpoint.

        Parameters
        ----------
        model : torch.nn.Module
            The model to be trained.

        Returns
        -------
        model : torch.nn.Module
            The model with loaded checkpoint weights.
        """
        # Resume training
        if self.args.resume:
            ckpt_file = ""
            if self.args.pretrained_weights is None:
                ckpt_file = os.path.join(self.file_name, "latest" + "_ckpt.pth")
            else:
                ckpt_file = self.args.ckpt

            if os.path.exists(ckpt_file):
                ckpt = torch.load(ckpt_file, map_location=self.device)
                model = load_ckpt(model, ckpt["model"])
                self.optimizer.load_state_dict(ckpt["optimizer"])
                self.best_aps = ckpt.pop(
                    "best_aps",
                    {
                        "best_average_ap50": 0,
                        "best_mAP50": 0,
                        "best_OKS50": 0,
                        "best_OKS95": 0,
                    },
                )
                start_epoch = self.args.start_epoch - 1 if self.args.start_epoch is not None else ckpt["start_epoch"]
                self.start_epoch = start_epoch
                logger.info("Loaded checkpoint '{}' (epoch {})".format(ckpt_file, self.start_epoch))  # noqa
        # Loading pretrained weights
        else:
            ckpt_file = ""
            if self.args.pretrained_weights is not None:
                ckpt_file = self.args.pretrained_weights
            elif os.path.exists(self.exp.pretrained_weights):
                ckpt_file = self.exp.pretrained_weights
            if ckpt_file:
                ckpt = torch.load(ckpt_file, map_location=self.device)["model"]
                model = load_ckpt(model, ckpt)
                logger.info("Loaded pretrained weights '{}'".format(ckpt_file))  # noqa

        self.start_epoch = 0

        return model

    def evaluate_and_save_model(self) -> None:
        """
        Evaluate the model on the validation set and save the latest and best checkpoint.

        Returns
        -------
        None
        """
        (performance, loss_meter), bb_detections, kp_detections = self.exp.eval(self.model, self.evaluator)

        best_average_ap50 = (performance["mAP"]["ap50"] + performance["OKS"]["kp_all"]["ap50"]) / 2

        if self.trial:
            self.trial.report(best_average_ap50, self.epoch)

            # Handle pruning based on the intermediate value.
            if self.trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        loss_str = ", ".join(["{}: {:.3f}".format(k, v.latest) for k, v in loss_meter.items()])

        update_best_ckpt = {
            "best_average_ap50": best_average_ap50 > self.best_aps["best_average_ap50"],
            "best_mAP50": performance["mAP"]["ap50"] > self.best_aps["best_mAP50"],
            "best_OKS50": performance["OKS"]["kp_all"]["ap50"] > self.best_aps["best_OKS50"],
            "best_OKS95": performance["OKS"]["kp_all"]["ap50_95"] > self.best_aps["best_OKS95"],
        }

        self.best_aps = {
            "best_average_ap50": max(self.best_aps["best_average_ap50"], best_average_ap50),
            "best_mAP50": max(self.best_aps["best_mAP50"], performance["mAP"]["ap50"]),
            "best_OKS50": max(self.best_aps["best_OKS50"], performance["OKS"]["kp_all"]["ap50"]),
            "best_OKS95": max(self.best_aps["best_OKS95"], performance["OKS"]["kp_all"]["ap50_95"]),
        }

        if self.args.logger == "tensorboard":
            for k, v in loss_meter.items():
                self.tblogger.add_scalar(f"train/{k}", v.latest, self.progress_in_iter)
            self.tblogger.add_scalar("val/COCOAP50", performance["mAP"]["ap50"], self.progress_in_iter)
            self.tblogger.add_scalar("val/COCOAP50_95", performance["mAP"]["ap50_95"], self.progress_in_iter)
            self.tblogger.add_scalar("val/OKS50", performance["OKS"]["kp_all"]["ap50"], self.progress_in_iter)
            self.tblogger.add_scalar("val/OKS50_95", performance["OKS"]["kp_all"]["ap50_95"], self.progress_in_iter)
        if self.args.logger == "wandb":
            metrics = {"val/loss/" + k: v.latest for k, v in loss_meter.items()}
            metrics["val/mAP/50"] = performance["mAP"]["ap50"]
            metrics["val/mAP/50_95"] = performance["mAP"]["ap50_95"]
            metrics["val/OKS_orig/50"] = performance["OKS"]["orig"]["ap50"]
            metrics["val/OKS_orig/50_95"] = performance["OKS"]["orig"]["ap50_95"]
            metrics["val/OKS_all/50"] = performance["OKS"]["kp_all"]["ap50"]
            metrics["val/OKS_all/50_95"] = performance["OKS"]["kp_all"]["ap50_95"]
            metrics["val/OKS_stem/50"] = performance["OKS"]["kp_stem"]["ap50"]
            metrics["val/OKS_stem/50_95"] = performance["OKS"]["kp_stem"]["ap50_95"]
            metrics["val/OKS_vein/50"] = performance["OKS"]["kp_vein"]["ap50"]
            metrics["val/OKS_vein/50_95"] = performance["OKS"]["kp_vein"]["ap50_95"]
            metrics["val/OKS_true/50"] = performance["OKS"]["kp_true"]["ap50"]
            metrics["val/OKS_true/50_95"] = performance["OKS"]["kp_true"]["ap50_95"]
            metrics["val/OKS_inbetween/50"] = performance["OKS"]["kp_inbetween"]["ap50"]
            metrics["val/OKS_inbetween/50_95"] = performance["OKS"]["kp_inbetween"]["ap50_95"]
            metrics["train/epoch"] = self.epoch + 1

            self.wandb_logger.log_metrics(metrics, step=self.progress_in_iter)
            if self.exp.debug:
                pass
                # self.wandb_logger.log_images(predictions)
        logger.info(f"Evaluation after epoch {self.epoch + 1}: " + loss_str)
        summary = (
            f'Average Precision (AP)\t\t@[ IoU=0.50:0.95\t] = {performance["mAP"]["ap50_95"]:.2f}'
            + f'\nAverage Precision (AP)\t\t@[ IoU=0.50\t\t] = {performance["mAP"]["ap50"]:.2f}'
        )
        summary += "\n"
        summary += (
            f"\nObject Keypoint Similarity (OKS) @[ IoU=0.50:0.95\t] = {performance['OKS']['kp_all']['ap50_95']:.2f}"
            + f"\nObject Keypoint Similarity (OKS) @[ IoU=0.5\t\t] = {performance['OKS']['kp_all']['ap50']:.2f}"
        )
        logger.info("\n" + summary)
        return update_best_ckpt

    def save_ckpt(self, ckpt_name: str, update_best_ckpt: dict, ap: float = 0.0) -> None:
        """
        Save the checkpoint.

        Parameters
        ----------
        ckpt_name : str
            The name of the checkpoint.
        update_best_ckpt : bool
            Whether to update the best checkpoint.
        ap : float
            The current AP.

        Returns
        -------
        None
        """
        save_model = self.model
        logger.info("Save weights to {}".format(self.file_name))
        ckpt_state = {
            "start_epoch": self.epoch + 1,
            "model": save_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "best_aps": self.best_aps,
            "curr_ap": ap,
        }
        save_checkpoint(
            ckpt_state,
            self.file_name,
            ckpt_name,
        )

        for k, v in update_best_ckpt.items():
            if v:
                save_checkpoint(
                    ckpt_state,
                    self.file_name,
                    k,
                )

        if self.args.logger == "wandb":
            self.wandb_logger.save_checkpoint(
                self.file_name,
                ckpt_name,
                update_best_ckpt,
                metadata={
                    "epoch": self.epoch + 1,
                    "optimizer": self.optimizer.state_dict(),
                    "best_aps": self.best_aps,
                    "curr_ap": ap,
                },
            )
