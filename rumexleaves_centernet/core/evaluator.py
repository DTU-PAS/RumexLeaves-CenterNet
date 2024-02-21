import contextlib
import io
import json
import os
import time
from itertools import chain, repeat

import cv2
import numpy as np
import pycocotools.coco as coco
import torch
from pycocotools.cocoeval import COCOeval

from rumexleaves_centernet.evaluation import dota_mAP
from rumexleaves_centernet.evaluation import compute_projected_oks
from rumexleaves_centernet.utils import ImageDrawer, MeterBuffer, ctdet_decode, kp_decode, obbox_2_box_corners


class Evaluator:
    def __init__(self, exp, args):
        self.exp = exp
        self.val_loader = self.exp.get_data_loader("val")
        self.norm_target = self.exp.norm_target
        self.down_ratio = self.exp.down_ratio
        self.output_size = (
            self.exp.input_size[0] // self.down_ratio,
            self.exp.input_size[1] // self.down_ratio,
        )
        self.device = exp.get_device()
        self.experiment_path = os.path.join(self.exp.output_dir, args.experiment_name)
        if self.exp.debug:
            self.debug_path = os.path.join(self.experiment_path, "debug")
            os.makedirs(self.debug_path, exist_ok=True)
            self.image_drawer = ImageDrawer(self.exp)
        self.dota_output_path = f"{self.experiment_path}/dota_eval"
        self.coco_output_path = f"{self.experiment_path}/coco_eval"
        self.meter = MeterBuffer(window_size=len(self.val_loader))
        self.generate_coco_annotations()
        if self.exp.target_mode_conf["do_obb"]:
            self.generate_dota_annotations()

    def generate_dota_annotations(self):
        os.makedirs(self.dota_output_path, exist_ok=True)
        dataset = self.val_loader.dataset

        img_lines = []
        for img_id in range(len(dataset)):
            img, target = dataset.get_img_bbtarget(img_id)
            bbs = target["bb"]["target"]
            lines = []
            for ann_id in range(bbs.shape[0]):
                obbox = bbs[ann_id, :]
                angle = obbox[4]
                width = obbox[2]
                height = obbox[3]
                cx, cy = obbox[0], obbox[1]
                tlbr = np.array([height / 2, width / 2, height / 2, width / 2])

                # Conver oriented bbox to straight bbox
                obbox_corners = obbox_2_box_corners(cx, cy, tlbr, angle)
                line = ""
                for corner in obbox_corners:
                    line += f"{corner[0]} {corner[1]} "
                line += f"{self.exp.classes[int(obbox[5])]} "
                line += "0"
                lines.append(line)
            img_str = (os.path.basename(target["img_id"])).replace(".png", ".txt")
            img_str = img_str.replace(".jpg", ".txt")
            img_lines.append(img_str.replace(".txt", ""))
            with open(f"{self.dota_output_path}/{img_str}", "w") as f:
                line_sep = repeat("\n")
                line_iter = chain.from_iterable(zip(lines, line_sep))
                f.writelines(line_iter)
        with open(f"{self.dota_output_path}/valset.txt", "w") as f:
            line_sep = repeat("\n")
            line_iter = chain.from_iterable(zip(img_lines, line_sep))
            f.writelines(line_iter)
        return

    def dota_eval(self, detections):
        dataset = self.val_loader.dataset
        detection_dict = {}
        for c in self.exp.classes:
            detection_dict[c] = []

        for img_id, detection in enumerate(detections):
            _, target = dataset.get_img_bbtarget(img_id)
            file_name = target["img_id"]
            img_str = (os.path.basename(file_name)).replace(".png", "")
            img_str = img_str.replace(".jpg", "")
            for obbox in detection:
                score = obbox[5]
                if score > self.exp.center_thresh:
                    label = self.exp.classes[int(obbox[-1])]
                    line_str = f"{img_str} {score} "
                    tlbr = np.array([obbox[3] / 2, obbox[2] / 2, obbox[3] / 2, obbox[2] / 2])

                    # Conver oriented bbox to straight bbox
                    obbox_corners = obbox_2_box_corners(obbox[0], obbox[1], tlbr, obbox[4])
                    for corner in obbox_corners:
                        line_str += f"{corner[0]} {corner[1]} "
                    detection_dict[label].append(line_str)

        for k, v in detection_dict.items():
            with open(f"{self.dota_output_path}/{k}.txt", "w") as f:
                line_sep = repeat("\n")
                line_iter = chain.from_iterable(zip(v, line_sep))
                f.writelines(line_iter)

        detpath = self.dota_output_path + "/{:s}.txt"
        annopath = self.dota_output_path + "/{:s}.txt"
        imagesetfile = f"{self.dota_output_path}/valset.txt"

        ap50, info = dota_mAP(detpath, annopath, imagesetfile, self.exp.classes)

        return ap50, ap50, info

    def generate_coco_annotations(self):
        os.makedirs(self.coco_output_path, exist_ok=True)

        dataset = self.val_loader.dataset

        categories = []
        for class_id, cat in enumerate(self.exp.classes):
            categories.append({"id": class_id, "name": cat})

        coco_anns = {"images": [], "annotations": [], "categories": categories}
        for img_id in range(len(dataset)):
            _, targets = dataset.get_img_bbtarget(img_id)
            file_name = targets["img_id"]
            bbs = targets["bb"]["target"]
            pl = targets["kp"]["target"]
            image_info = {"file_name": file_name, "id": img_id}
            coco_anns["images"].append(image_info)
            for ann_id in range(bbs.shape[0]):
                bbox = bbs[ann_id, :]
                cat_id = int(bbox[5])
                width = bbox[2]
                height = bbox[3]
                tlbr = np.array([height / 2, width / 2, height / 2, width / 2])

                # Convert oriented bbox to straight bbox
                bbox_corner = obbox_2_box_corners(bbox[0], bbox[1], tlbr, bbox[4])
                x_1 = np.min(bbox_corner[:, 0])
                y_1 = np.min(bbox_corner[:, 1])
                x_2 = np.max(bbox_corner[:, 0])
                y_2 = np.max(bbox_corner[:, 1])
                bbox = [int(x_1), int(y_1), int(x_2 - x_1), int(y_2 - x_1)]

                kp = []
                pl_i = pl[ann_id]
                if len(pl_i) > 0:
                    pl_i = np.concatenate((pl_i, np.ones((pl_i.shape[0], 1)) * 2), axis=1)
                    kp = pl_i.reshape(-1).tolist()

                annotation_info = {
                    "image_id": img_id,
                    "id": int(len(coco_anns["annotations"])),
                    "category_id": cat_id,
                    "bbox": bbox,
                    "area": int(width * height),
                    "iscrowd": 0,
                    "keypoints": kp,
                    "num_keypoints": int(len(kp) / 3),
                }
                coco_anns["annotations"].append(annotation_info)

        json.dump(coco_anns, open(f"{self.coco_output_path}/instances_val.json", "w"))

    def coco_pytool_eval(self, bb_detections, kp_detections, projected=False, mode="kp_all"):
        coco_dets = []
        for img_id in range(len(bb_detections), len(kp_detections)):
            if len(bb_detections) > 0:
                for bb in bb_detections[img_id]:
                    if bb[4] > self.exp.center_thresh:
                        coco_det = {
                            "image_id": img_id,
                            "category_id": int(bb[5]),
                            "bbox": [
                                int(bb[0]),
                                int(bb[1]),
                                int(bb[2] - bb[0]),
                                int(bb[3] - bb[1]),
                            ],
                            "score": float("{:.2f}".format(bb[4])),
                        }
                    coco_dets.append(coco_det)
            if len(kp_detections) > 0:
                for kp_detection in kp_detections[img_id]:
                    score = kp_detection[0][2]
                    class_id = kp_detection[0][3]
                    if score > self.exp.center_thresh:
                        kp_detection = kp_detection[:, :2]
                        kp_detection = np.concatenate(
                            (kp_detection, np.ones((kp_detection.shape[0], 1)) * 1),
                            axis=1,
                        )
                        kp_detection = kp_detection.reshape(-1).tolist()
                        coco_det = {
                            "image_id": img_id,
                            "category_id": int(class_id),
                            "keypoints": kp_detection,
                            "score": float("{:.2f}".format(score)),
                        }
                        coco_dets.append(coco_det)

        if len(coco_dets) == 0:
            return 0, 0, ""
        json.dump(coco_dets, open("{}/results.json".format(self.coco_output_path), "w"))

        cocoGt = coco.COCO(self.coco_output_path + "/instances_val.json")
        coco_dets = cocoGt.loadRes("{}/results.json".format(self.coco_output_path))
        if len(bb_detections) > 0:
            coco_eval = COCOeval(cocoGt, coco_dets, "bbox")
            coco_eval.evaluate()
            coco_eval.accumulate()
            redirect_string = io.StringIO()
            with contextlib.redirect_stdout(redirect_string):
                coco_eval.summarize()
            info = redirect_string.getvalue()

        if len(kp_detections) > 0:
            coco_eval = COCOeval(cocoGt, coco_dets, "keypoints")
            if projected:

                def func(self, imgId, catId):
                    return compute_projected_oks(self, imgId, catId, mode)

                coco_eval.computeOks = func.__get__(coco_eval, COCOeval)
            else:
                sigmas = [0.05 for i in range(int(kp_detections[0].shape[1]))]
                coco_eval.params.kpt_oks_sigmas = np.array(sigmas)
            coco_eval.evaluate()
            coco_eval.accumulate()
            redirect_string = io.StringIO()
            with contextlib.redirect_stdout(redirect_string):
                coco_eval.summarize()
            info = redirect_string.getvalue()
        ap50_95, ap50 = coco_eval.stats[0], coco_eval.stats[1]

        return ap50_95, ap50, info

    def before_eval(self, model):
        model.eval()
        self.meter.clear_meters()

    def evaluate(self, model):
        self.before_eval(model)
        bb_detections = []
        kp_detections = []

        for iter, batch in enumerate(self.val_loader):
            iter_bb_detections, iter_kp_detections = self.eval_iter(model, iter, batch)
            bb_detections.extend(iter_bb_detections)
            kp_detections.extend(iter_kp_detections)

        return self.after_eval(bb_detections, kp_detections)

    def denorm(self, predictions, targets):
        if "obb" in predictions:
            denorm_values = [self.output_size[0], self.output_size[1]]
            if self.exp.target_mode_conf["box_mode"] == "tlbr":
                denorm_values.extend([self.output_size[0], self.output_size[1]])
            if self.exp.target_mode_conf["angle_mode"] == "360":
                denorm_values.append(360.0)
            elif self.exp.target_mode_conf["angle_mode"] == "sincos":
                denorm_values.extend([1.0, 1.0])
            denorm = torch.tensor(denorm_values).to(self.device)
            targets["obb"] *= denorm
            predictions["obb"] *= denorm.unsqueeze(1).unsqueeze(1).expand(predictions["obb"].shape)
        if "kp" in predictions:
            denorm_values_dist = torch.ones((8, 1)) * self.output_size[0]
            if self.exp.target_mode_conf["angle_mode"] == "360":
                denorm_values_angle = [torch.ones((8, 1)) * 360]
            elif self.exp.target_mode_conf["angle_mode"] == "sincos":
                denorm_values_angle = [torch.ones((8, 1)), torch.ones((8, 1))]
            mult = torch.cat([denorm_values_dist, *denorm_values_angle], dim=1).reshape(-1).to(self.device)
            targets["kp"] *= mult
            predictions["kp"] *= mult.unsqueeze(1).unsqueeze(1)
        return predictions, targets

    def eval_iter(self, model, iter, batch):
        iter_start_time = time.time()

        targets = {k: batch[k] for k in batch.keys() if k != "img" and k != "meta"}

        for k in targets:
            targets[k] = targets[k].to(self.device)
        imgs = batch["img"].to(self.device)
        data_end_time = time.time()

        # forward
        predictions, loss_stats = model(imgs, targets)

        inference_end_time = time.time()

        # decode
        if self.norm_target:
            predictions, targets = self.denorm(predictions, targets)

        kps = []
        if "kp" in predictions.keys():
            if self.exp.target_mode_conf["do_kphm"]:
                kphm = predictions["kphm"]
            else:
                kphm = None

            kps = kp_decode(
                predictions["hm"],
                predictions["kp"],
                self.exp.target_mode_conf["angle_mode"],
                kphm=kphm,
                reg=predictions["off"],
            )
            kps[:, :, :, :2] *= self.down_ratio

        dets = torch.empty((0, 0)).to(self.device)
        if "obb" in predictions.keys():
            dets = ctdet_decode(
                predictions["hm"],
                predictions["obb"],
                self.exp.target_mode_conf,
                reg=predictions["off"],
            )
            dets[:, :, :4] *= self.down_ratio

        iter_end_time = time.time()

        # logging
        self.meter.update(
            iter_time=iter_end_time - iter_start_time,
            inference_time=inference_end_time - data_end_time,
            data_time=data_end_time - iter_start_time,
            **loss_stats,
        )

        # debugging
        if self.exp.debug:
            if self.exp.target_mode_conf["angle_mode"] == "sincos":
                if self.exp.target_mode_conf["do_kp"]:
                    b, n, _ = targets["kp"].shape
                    targets["kp"] = targets["kp"].reshape(-1, 8, 3)
                    targets["kp"][:, :, 1] = (
                        torch.arctan2(((targets["kp"][:, :, 2] * 2) - 1), ((targets["kp"][:, :, 1] * 2) - 1))
                        * 180
                        / torch.pi
                    )
                    targets["kp"] = targets["kp"][:, :, :2]
                    targets["kp"] = targets["kp"].reshape((b, n, -1))

                if self.exp.target_mode_conf["do_obb"]:
                    if self.exp.target_mode_conf["box_mode"] == "tlbr":
                        v = 4
                    elif self.exp.target_mode_conf["box_mode"] == "wh":
                        v = 2
                    targets["obb"][:, :, v] = (
                        torch.arctan2(((targets["obb"][:, :, v + 1] * 2) - 1), ((targets["obb"][:, :, v] * 2) - 1))
                        * 180
                        / torch.pi
                    )
            self.imwrite_debug_images(iter, imgs, targets, predictions, dets, kps, batch["meta"])

        if torch.is_tensor(dets):
            dets = dets.detach().cpu().numpy()
        if torch.is_tensor(kps):
            kps = kps.detach().cpu().numpy()

        return dets, kps

    def after_eval(self, bb_detections, kp_detections):
        loss_meter = self.meter.get_filtered_meter("loss")

        performance = {
            "mAP": {"ap50": 0, "ap50_95": 0},
            "OKS": {
                "orig": {"ap50": 0, "ap50_95": 0},
                "kp_all": {"ap50": 0, "ap50_95": 0},
                "kp_stem": {"ap50": 0, "ap50_95": 0},
                "kp_vein": {"ap50": 0, "ap50_95": 0},
                "kp_true": {"ap50": 0, "ap50_95": 0},
                "kp_inbetween": {"ap50": 0, "ap50_95": 0},
            },
        }

        oks_50_95, oks_50 = 0, 0
        ap50_95, ap50 = 0, 0
        if "kp" in self.exp.heads:
            # OKS: kp_all evaluation
            oks_50_95, oks_50, _ = self.coco_pytool_eval([], kp_detections, projected=False)
            performance["OKS"]["orig"]["ap50_95"] = oks_50_95
            performance["OKS"]["orig"]["ap50"] = oks_50
            # projected OKS: kp_all evaluation
            oks_50_95, oks_50, _ = self.coco_pytool_eval([], kp_detections, projected=True, mode="kp_all")
            performance["OKS"]["kp_all"]["ap50_95"] = oks_50_95
            performance["OKS"]["kp_all"]["ap50"] = oks_50
            # projected OKS: kp_stem evaluation
            oks_50_95, oks_50, _ = self.coco_pytool_eval([], kp_detections, projected=True, mode="kp_stem")
            performance["OKS"]["kp_stem"]["ap50_95"] = oks_50_95
            performance["OKS"]["kp_stem"]["ap50"] = oks_50
            # projected OKS: kp_vein evaluation
            oks_50_95, oks_50, _ = self.coco_pytool_eval([], kp_detections, projected=True, mode="kp_vein")
            performance["OKS"]["kp_vein"]["ap50_95"] = oks_50_95
            performance["OKS"]["kp_vein"]["ap50"] = oks_50
            # projected OKS: kp_true evaluation
            oks_50_95, oks_50, _ = self.coco_pytool_eval([], kp_detections, projected=True, mode="kp_true")
            performance["OKS"]["kp_true"]["ap50_95"] = oks_50_95
            performance["OKS"]["kp_true"]["ap50"] = oks_50
            # projected OKS: kp_inbetween evaluation
            oks_50_95, oks_50, _ = self.coco_pytool_eval([], kp_detections, projected=True, mode="kp_inbetween")
            performance["OKS"]["kp_inbetween"]["ap50_95"] = oks_50_95
            performance["OKS"]["kp_inbetween"]["ap50"] = oks_50

        if "obb" in self.exp.heads:
            ap50_95, ap50, _ = self.dota_eval(bb_detections)
            performance["mAP"]["ap50_95"] = ap50_95
            performance["mAP"]["ap50"] = ap50
        return (performance, loss_meter), bb_detections, kp_detections

    def imwrite_debug_images(self, iter, imgs, targets, predictions, dets, kps, meta):
        for k in range(imgs.shape[0]):
            # Get kth datapoint and its prediction
            img = imgs[k, :, :, :]
            if self.exp.p_normalize:
                img = ImageDrawer.tensor_to_cv2_img(img, ds_mean=self.exp.ds_mean, ds_std=self.exp.ds_std)
            else:
                img = ImageDrawer.tensor_to_cv2_img(img)
            target = {key: targets[key][k] for key in targets.keys()}
            prediction = {key: predictions[key][k] for key in predictions.keys()}
            if len(dets) > 0:
                det = dets[k, :, :]
            if len(kps) > 0:
                kp = kps[k, :, :, :]

            # Generate ground truth and prediction image
            gt_img = img.copy()
            if "obb" in target.keys():
                gt_img = self.image_drawer.draw_ground_truth_bb(gt_img, target)
            if "kp" in target.keys():
                gt_img = self.image_drawer.draw_gt_kps(gt_img, target)
            gt_hm = self.image_drawer.draw_heat_map(img.copy(), target["hm"])
            pred_img = img.copy()
            if len(dets) > 0:
                pred_img = self.image_drawer.draw_predicted_bb(pred_img, det)
            if len(kps) > 0:
                pred_img = self.image_drawer.draw_pred_kps(pred_img, kp)
            pred_hm = self.image_drawer.draw_heat_map(img.copy(), prediction["hm"])
            image_row_1 = cv2.hconcat([gt_img, gt_hm])
            image_row_2 = cv2.hconcat([pred_img, pred_hm])
            combined_image = cv2.vconcat([image_row_1, image_row_2])
            cv2.imwrite(os.path.join(self.debug_path, f"sample_{iter}_{k}.jpg"), combined_image)
            os.makedirs(os.path.join(self.debug_path, "predictions"), exist_ok=True)
            pred_img = cv2.cvtColor(pred_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(self.debug_path, "paper_imgs", f"{os.path.basename(meta['img_id'][k])}"), pred_img)
