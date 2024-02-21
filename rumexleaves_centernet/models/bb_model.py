import torch
import torch.nn as nn

from rumexleaves_centernet.models.dcn_resnet import BasicBlock, Bottleneck, PoseResNet
from rumexleaves_centernet.models.losses import FocalLoss, RegL1Loss, RegLoss, VeinStemLoss
from rumexleaves_centernet.utils.model_utils import _sigmoid

resnet_spec = {
    18: (BasicBlock, [2, 2, 2, 2]),
    34: (BasicBlock, [3, 4, 6, 3]),
    50: (Bottleneck, [3, 4, 6, 3]),
    101: (Bottleneck, [3, 4, 23, 3]),
    152: (Bottleneck, [3, 8, 36, 3]),
}


class BBModel(nn.Module):
    def __init__(self, exp):
        super().__init__()
        self.exp = exp
        # model_architecture
        model_architecture = PoseResNet

        self.model = self.make_model(
            model_architecture,
            self.exp.num_layers,
            self.exp.heads,
            self.exp.head_conv,
            self.exp.head_complexity,
            self.exp.up_dc,
            True,
            self.exp.init_bias,
        )
        self.center_loss = (
            FocalLoss()
            if self.exp.center_loss == "focal"
            else torch.nn.MSELoss()
            if self.exp.center_loss == "mse"
            else None
        )
        self.regression_loss = (
            RegL1Loss()
            if self.exp.regression_loss == "l1"
            else RegLoss()
            if self.exp.regression_loss == "sl1"
            else None
        )

        sincos = False
        if self.exp.target_mode_conf == "sincos":
            sincos = True
        self.vein_loss = VeinStemLoss(self.exp.norm_target, sincos) if self.exp.vein_loss else self.regression_loss

        self.loss_weights = {"hm": 1.0, "obb": 1.0, "off": 1.0, "kp": 0.0, "angle": 0.0, "kphm": 0.0}
        for k, v in self.exp.loss_weights.items():
            if k in self.loss_weights.keys():
                self.loss_weights[k] = v
        self.kp_weights = {
            "dist": 1.0,
            "angle": 1.0,
        }
        for k, v in self.exp.kp_weights.items():
            if k in self.kp_weights.keys():
                self.kp_weights[k] = v

        # Normalize kp weights
        sum_kp_weights = sum(self.kp_weights.values())
        self.kp_weights["dist"] /= sum_kp_weights
        self.kp_weights["angle"] /= sum_kp_weights

    @staticmethod
    def make_model(model_arch, num_layers, heads, head_conv, head_complexity, up_dc, pretrained, init_bias=None):
        block_class, layers = resnet_spec[num_layers]

        model = model_arch(
            block_class, layers, heads, head_conv=head_conv, head_complexity=head_complexity, up_dc=up_dc
        )
        model.init_weights(num_layers, init_bias, pretrained=pretrained)
        return model

    def forward(self, x, targets=None):
        # fpn output content features of [dark3, dark4, dark5]
        predictions = self.model(x)[0]
        if type(self.center_loss) is not torch.nn.MSELoss:
            predictions["hm"] = _sigmoid(predictions["hm"])
            if "kphm" in predictions.keys():
                predictions["kphm"] = _sigmoid(predictions["kphm"])

        if targets:
            hm_loss = self.center_loss(predictions["hm"], targets["hm"])

            hm_kp_loss = 0
            if self.loss_weights["kphm"] > 0:
                hm_kp_loss = self.center_loss(predictions["kphm"], targets["kphm"])

            wh_loss = 0
            if self.loss_weights["obb"] > 0:
                wh_loss = self.regression_loss(
                    predictions["obb"],
                    targets["reg_mask"],
                    targets["ind"],
                    targets["obb"],
                )

            off_loss = 0
            if self.loss_weights["off"] > 0:
                off_loss = self.regression_loss(
                    predictions["off"],
                    targets["reg_mask"],
                    targets["ind"],
                    targets["off"],
                )

            kp_loss = 0
            if self.loss_weights["kp"] > 0:
                kp_loss = self.vein_loss(
                    predictions["kp"],
                    targets["kp_reg_mask"],
                    targets["ind"],
                    targets["kp"],
                )
            angle_loss = 0
            if self.loss_weights["angle"] > 0:
                angle_loss = self.regression_loss(
                    predictions["angle"],
                    targets["reg_mask"],
                    targets["ind"],
                    targets["angle"],
                )
            loss = (
                self.loss_weights["hm"] * hm_loss
                + self.loss_weights["obb"] * wh_loss
                + self.loss_weights["off"] * off_loss
                + self.loss_weights["kp"] * kp_loss
                + self.loss_weights["angle"] * angle_loss
                + self.loss_weights["kphm"] * hm_kp_loss
            )
            loss_stats = {
                "loss": loss,
                "hm_loss": hm_loss,
                "wh_loss": wh_loss,
                "off_loss": off_loss,
                "kp_loss": kp_loss,
                "angle_loss": angle_loss,
                "kp_hm_loss": hm_kp_loss,
            }
        else:
            loss_stats = None

        return predictions, loss_stats
