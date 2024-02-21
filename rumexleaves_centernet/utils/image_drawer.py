import math

import cv2
import numpy as np
from torchvision import transforms

from rumexleaves_centernet.utils import obbox_2_box_corners


class ImageDrawer:
    def __init__(self, exp) -> None:
        self.exp = exp
        self.down_ratio = self.exp.down_ratio
        self.center_thresh = self.exp.center_thresh
        self.classes = self.exp.classes
        self.colors = {"bbox": (255, 255, 0), "kp": (255, 0, 0)}

    @staticmethod
    def tensor_to_cv2_img(img, ds_mean=None, ds_std=None):
        if ds_mean and ds_std:
            inv_std = [
                1 / ds_std[0],
                1 / ds_std[1],
                1 / ds_std[2],
            ]
            inv_mean = [
                -ds_mean[0],
                -ds_mean[1],
                -ds_mean[2],
            ]
            inv_trans = transforms.Compose(
                [
                    transforms.Normalize(mean=[0.0, 0.0, 0.0], std=inv_std),
                    transforms.Normalize(mean=inv_mean, std=[1.0, 1.0, 1.0]),
                ]
            )
            img = inv_trans(img)
        img = img.cpu().numpy().transpose(1, 2, 0)
        img = (img * 255).astype(np.uint8)
        return img

    def draw_gt_kps(self, img, target):
        img_h, img_w = img.shape[:2]
        for index, offset, kp in zip(target["ind"], target["off"], target["kp"]):
            index = index.cpu().numpy()
            kp = kp.reshape(-1, 2)
            kp = kp.cpu().numpy()
            offset = offset.cpu().numpy()
            if index == 0:
                break
            cx = (index % (img_w // self.down_ratio) + offset[0]) * self.down_ratio
            cy = (math.floor(index // (img_w // self.down_ratio)) + offset[1]) * self.down_ratio
            for i, point in enumerate(kp):
                kp[i] = [
                    cx + math.cos(point[1] * math.pi / 180) * point[0] * self.down_ratio,
                    cy + math.sin(point[1] * math.pi / 180) * point[0] * self.down_ratio,
                ]
                cv2.circle(img, (int(kp[i][0]), int(kp[i][1])), 2, self.colors["kp"], 2)
            cv2.polylines(img, [kp.astype(int)], 0, self.colors["kp"], 2)
        return img

    def draw_pred_kps(self, img, kps):
        kps = kps.detach().cpu().numpy()
        scores = kps[:, :, 2]
        stems = kps[:, :, :2]
        for score, stem in zip(scores, stems):
            if score[0] > self.center_thresh:
                for stem_point in stem:
                    cv2.circle(img, (int(stem_point[0]), int(stem_point[1])), 2, self.colors["kp"], 7)
                cv2.polylines(img, [stem.astype(int)], 0, self.colors["kp"], 2)
        return img

    def draw_ground_truth_bb(self, img, target):
        img_h, img_w = img.shape[:2]

        for i, (index, offset, bb) in enumerate(zip(target["ind"], target["off"], target["obb"])):
            index = index.cpu().numpy()
            bb = bb.cpu().numpy()
            offset = offset.cpu().numpy()
            if index == 0:
                break
            cx = (index % (img_w // self.down_ratio) + offset[0]) * self.down_ratio
            cy = (math.floor(index // (img_w // self.down_ratio)) + offset[1]) * self.down_ratio
            angle = 0.0

            if self.exp.target_mode_conf["box_mode"] == "wh":
                bb[:2] *= self.down_ratio
                angle = bb[2]
                tlbr = np.array([bb[0] / 2, bb[1] / 2, bb[0] / 2, bb[1] / 2])

            elif self.exp.target_mode_conf["box_mode"] == "tlbr":
                bb[:4] *= self.down_ratio
                angle = bb[4]
                tlbr = bb[:4]

            bb_corners = obbox_2_box_corners(cx, cy, tlbr, angle)

            cv2.circle(img, (round(cx), round(cy)), 5, self.colors["bbox"], 2)
            cv2.drawContours(img, [bb_corners.astype(int)], 0, self.colors["bbox"], 2)
        return img

    def _draw_predicted_bb(self, img, bb, cat, conf=1, show_txt=True):
        bb = bb.detach().cpu().numpy()
        bb = np.array(bb, dtype=np.int32)
        cat = int(cat)
        txt = "{}{:.1f}".format(self.classes[cat], conf)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cat_size = cv2.getTextSize(txt, font, 0.5, 2)[0]
        x1, y1, x2, y2 = (
            bb[0] - bb[2] // 2,
            bb[1] - bb[3] // 2,
            bb[0] + bb[2] // 2,
            bb[1] + bb[3] // 2,
        )

        cv2.rectangle(img, (x1, y1), (x2, y2), self.colors["bbox"], 2)
        cv2.circle(img, (round(x1 + (x2 - x1) / 2), round(y1 + (y2 - y1) / 2)), 5, self.colors["bbox"], 2)

        if show_txt:
            cv2.rectangle(img, (x1, y1 - cat_size[1] - 2), (x1 + cat_size[0], y1 - 2), self.colors["bbox"], -1)
            cv2.putText(
                img,
                txt,
                (x1, y1 - 2),
                font,
                0.5,
                (0, 0, 0),
                thickness=1,
                lineType=cv2.LINE_AA,
            )
        return img

    def _draw_predicted_bb_oriented(self, img, bb, angle, cat, conf, show_txt=True):
        bb = bb.detach().cpu().numpy()
        bb = np.array(bb, dtype=np.int32)
        angle = angle.detach().cpu().item()
        width = bb[2]
        height = bb[3]
        cx, cy = bb[0], bb[1]
        tlbr = np.array([height / 2, width / 2, height / 2, width / 2])

        # Conver oriented bbox to straight bbox
        bb_corners = obbox_2_box_corners(cx, cy, tlbr, angle)
        cat = int(cat)
        txt = "{:.1f}".format(conf)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cat_size = cv2.getTextSize(txt, font, 0.5, 5)[0]
        cv2.drawContours(img, [bb_corners.astype(int)], 0, self.colors["bbox"], 5)
        if show_txt:
            cat_img = np.zeros(img.shape, dtype=np.uint8)
            x1, y1 = bb[0] - bb[2] // 2, bb[1] - bb[3] // 2
            cv2.rectangle(cat_img, (x1, y1 - cat_size[1] - 2), (x1 + cat_size[0], y1 - 2), self.colors["bbox"], -1)
            cv2.putText(
                cat_img,
                txt,
                (x1, y1 - 2),
                font,
                0.5,
                (10, 10, 10),
                thickness=1,
                lineType=cv2.LINE_AA,
            )
            rot_mat = cv2.getRotationMatrix2D([float(cx), float(cy)], -angle, 1.0)
            cat_img = cv2.warpAffine(cat_img, rot_mat, cat_img.shape[1::-1], flags=cv2.INTER_LINEAR)
            img[cat_img[:, :, 0] > 0] = cat_img[cat_img[:, :, 0] > 0]

        return img

    def draw_predicted_bb(self, img, det):
        # ToDo: support num_classes > 1
        for d in det:
            if d[5] > self.center_thresh:
                img = self._draw_predicted_bb_oriented(img, d[:4], d[4], d[-1], d[5])
        return img

    def draw_heat_map(self, img, heat_map):
        # ToDo: support num_classes > 1
        img_h, img_w = img.shape[:2]

        img_hm = heat_map.detach().cpu().numpy()
        img_hm = (img_hm[0, :, :] * 255).astype(np.uint8).copy()
        img_hm = cv2.resize(img_hm, (img_w, img_h))
        # convert gray to RGB
        img_hm = cv2.cvtColor(img_hm, cv2.COLOR_GRAY2RGB)
        return img_hm
