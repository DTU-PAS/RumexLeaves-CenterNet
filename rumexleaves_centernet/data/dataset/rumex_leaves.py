#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import copy
import os
import os.path

import cv2
import numpy as np
from annotation_converter.AnnotationConverter import AnnotationConverter
from torch.utils.data import Dataset

from rumexleaves_centernet.data.dataset.target_reformulate import reformulate_target


class RumexLeavesDataset(Dataset):

    """
    Close-up leave data from RumexWeed Dataset

    input is image, target is annotation

    Args:
        data_dir (string): filepath to RumexWeeds folder.
        image_list (list(string)): list of img ids to consider
        image_size (tuple(int, int)): image size
        preproc (callable, optional): transformation to perform on the
            input image
    """

    def __init__(
        self,
        data_dir,
        image_list,
        classes,
        target_mode_conf,
        preproc=None,
        norm_target=False,
        cp_i=2,
        annotation_file_rel_to_img="../../annotations_bb.xml",
    ):
        super().__init__()
        self.data_dir = data_dir
        self.image_list = image_list
        self.classes = classes
        self.target_mode_conf = target_mode_conf
        if self.target_mode_conf["box_mode"] == "wh":
            self.target_mode_conf["cp_i"] = -1
        self.cp_i = cp_i
        self.preproc = preproc
        self.norm_target = norm_target
        self.imgs = None
        self.annotation_file_rel_to_img = annotation_file_rel_to_img
        self.annotations = self._load_annotations()

    def __len__(self):
        return len(self.image_list)

    def _get_img_ids(self):
        ids = []
        for img_path in self.image_list:
            ids.append(os.path.basename(img_path))
        return ids

    def _load_annotations(self):
        return [self.load_annotation_from_id(_ids) for _ids in self.image_list]

    def load_annotation_from_id(self, id_):
        annotation_file = f"{os.path.dirname(f'{self.data_dir}/{id_}')}/{self.annotation_file_rel_to_img}"
        img_annotation = AnnotationConverter.read_cvat_by_id(annotation_file, os.path.basename(id_))
        if img_annotation is None:
            print(f"Image file {id_} not found")
            return None
        img_width, img_height = int(img_annotation.get_img_width()), int(img_annotation.get_img_height())
        target = {}
        target["bb"] = self.load_bb_targets(img_annotation)
        target["kp"] = self.load_keypoint_targets(img_annotation, target["bb"])
        target["img_id"] = id_
        target["img_info"] = {"orig_size": (img_height, img_width)}
        return target

    def load_bb_targets(self, img_annotation):
        bbs = img_annotation.get_bounding_boxes()
        targets = np.zeros((0, 6))
        for i, bb in enumerate(bbs):
            label = bb.get_label()
            if label in self.classes:
                obj_id = self.classes.index(bb.get_label())
            else:
                continue
            cx, cy, w, h = bb.get_xywh()
            angle = bb.get_rotation() * 180 / np.pi
            bb_t = [cx, cy, w, h, angle, obj_id]
            targets = np.append(targets, [bb_t], axis=0)
        return targets

    def load_keypoint_targets(self, img_annotation, bb_targets):
        polylines = img_annotation.get_polylines()
        num_keypoints = 8
        targets = np.zeros((len(polylines), num_keypoints, 2))
        for i, bb in enumerate(bb_targets):
            matching_kpoints = []
            dist = 100000
            for kp in polylines:
                points = kp.get_polyline_points_as_array()
                mean = [np.mean(points[-5:, 0]), np.mean(points[-5:, 1])]
                kp_dist = np.linalg.norm(mean - bb[:2])
                if kp_dist < dist:
                    matching_kpoints = points
                    dist = kp_dist
            if len(matching_kpoints) < num_keypoints:
                matching_kpoints = np.transpose(
                    (
                        np.array(
                            [
                                np.pad(
                                    matching_kpoints[:, 0],
                                    (num_keypoints - len(matching_kpoints), 0),
                                    "edge",
                                ),
                                np.pad(
                                    matching_kpoints[:, 1],
                                    (num_keypoints - len(matching_kpoints), 0),
                                    "edge",
                                ),
                            ]
                        )
                    )
                )
            targets[i, :, :] = matching_kpoints
        return targets

    def load_image(self, index):
        img_id = self.annotations[index]["img_id"]

        img_file = os.path.join(self.data_dir, img_id)

        img = cv2.imread(img_file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        assert img is not None

        return img

    def pull_item(self, index):
        target = self.annotations[index]
        img = self.load_image(index)
        return img, copy.deepcopy(target)

    def get_img_bbtarget(self, index):
        img, target = self.pull_item(index)
        if self.preproc is not None:
            img, target = self.preproc(img, target)
        return img, target

    def __getitem__(self, index):
        img, target = self.get_img_bbtarget(index)
        d_point = reformulate_target(img, len(self.classes), target, self.target_mode_conf)
        d_point["meta"] = {"img_id": target["img_id"]}
        return d_point
