from abc import ABCMeta

import albumentations as A
import albumentations.pytorch as AP
import cv2
import numpy as np


class Transform(metaclass=ABCMeta):
    def __call__(self, image, target):
        kp_targets = target["kp"]
        bb_targets = target["bb"]
        kp_targets_in = np.zeros((kp_targets.shape[0], kp_targets.shape[1], 2))
        kp_targets_in = np.c_[kp_targets, kp_targets_in]
        kp_info = np.zeros((kp_targets_in.shape[0], kp_targets_in.shape[1], 1))
        kp_info[:, :, 0] = -1 * np.arange(1, kp_info.shape[1] + 1)

        bb_targets_in = np.ones((len(bb_targets), 4))
        bb_targets_in[:, :2] = bb_targets[:, :2]
        bb_targets_in[:, 2] = bb_targets[:, 4]
        bb_targets_in = np.expand_dims(bb_targets_in, 1)

        all_targets_in = np.concatenate((bb_targets_in, kp_targets_in), axis=1)

        info = np.zeros((all_targets_in.shape[0], all_targets_in.shape[1], 3))
        info[:, 0, 0] = bb_targets[:, 2]  # width
        info[:, 0, 1] = bb_targets[:, 3]  # height
        info[:, 0, 2] = bb_targets[:, 5]  # class
        info[:, 1:, 2] = kp_info.squeeze()

        all_targets_in = np.reshape(all_targets_in, (-1, all_targets_in.shape[-1]))
        info = np.reshape(info, (-1, info.shape[-1]))
        transformed = self.albu_transform(
            image=image,
            keypoints=all_targets_in,
            additional_info=info,
            bboxes=[],
            category_ids=[],
        )
        image = transformed["image"]

        transformed_targets = np.array(transformed["keypoints"])
        info = np.array(transformed["additional_info"])

        # to bb_target format
        transformed_bb_targets = transformed_targets[info[:, 2] >= 0, :]
        bb_visibility = np.array(
            [
                not (i[0] < 0 or i[1] < 0 or i[0] >= image.shape[1] or i[1] >= image.shape[2])
                for i in transformed_bb_targets
            ]
        )

        info_bb = info[info[:, 2] >= 0, :]
        num_targets = len(transformed_bb_targets)
        if num_targets == 0:
            bb_targets_out = np.zeros((0, 6))
        else:
            bb_targets_out = np.zeros((num_targets, 6))
            bb_targets_out[:, :2] = transformed_bb_targets[:, :2]
            bb_targets_out[:, 2] = info_bb[:, 0] * transformed_bb_targets[:, 3]
            bb_targets_out[:, 3] = info_bb[:, 1] * transformed_bb_targets[:, 3]
            bb_targets_out[:, 4] = transformed_bb_targets[:, 2]
            bb_targets_out[:, 5] = info_bb[:, 2]

        # to keypoint_target format
        transformed_kp_targets = transformed_targets[info[:, 2] < 0, :]
        kp_visibility = np.array(
            [
                not (i[0] < 0 or i[1] < 0 or i[0] >= image.shape[1] or i[1] >= image.shape[2])
                for i in transformed_kp_targets
            ]
        )
        num_targets = len(transformed_kp_targets)
        if num_targets == 0:
            kp_targets_out = np.zeros((0, 8, 2))
        else:
            kp_targets_out = transformed_kp_targets[:, :2].reshape(kp_targets.shape)
            kp_visibility = kp_visibility.reshape((kp_targets.shape[0], kp_targets.shape[1], 1))

        target["bb"] = {"target": bb_targets_out, "vis": bb_visibility}
        target["kp"] = {"target": kp_targets_out, "vis": kp_visibility}
        return image, target


class TrainTransform(Transform):
    def __init__(
        self,
        max_labels=50,
        image_size=(640, 640),
        p_scale_shift=0.0,
        p_flip=0.0,
        p_rotate=0.0,
        p_color_jitter=0.0,
        jitter_values=[0.2, 0.2, 0.2, 0.2],
        p_rgb_shift=0.0,
        rgb_shift_values=[50, 50, 50],
        p_blur=0.0,
        p_random_brightness_contrast=0.0,
        p_normalize=0.0,
        ds_mean=(0.32318847, 0.3228973, 0.1618311),
        ds_std=(0.14185149, 0.13611854, 0.14185149, 0.10307392),
    ):
        self.max_labels = max_labels
        self.image_size = image_size

        augmentation_list = []
        augmentation_list.append(A.LongestMaxSize(max_size=image_size[0], interpolation=1, p=1.0))
        augmentation_list.append(
            A.PadIfNeeded(
                min_height=image_size[0],
                min_width=image_size[1],
                border_mode=0,
                position="center",
                value=None,
                mask_value=None,
                p=1.0,
            )
        )

        # For now we are only zooming out.
        augmentation_list.append(
            A.ShiftScaleRotate(
                shift_limit=0.0,
                scale_limit=[-0.5, 0.0],
                rotate_limit=0,
                border_mode=cv2.BORDER_CONSTANT,
                p=p_scale_shift,
            )
        )

        if p_flip > 0:
            augmentation_list.append(A.HorizontalFlip(p=p_flip))
            augmentation_list.append(A.VerticalFlip(p=p_flip))
        if p_rotate > 0:
            augmentation_list.append(A.RandomRotate90(p=p_rotate))
            # augmentation_list.append(A.Rotate(limit=180, p=p_rotate))
        if p_color_jitter > 0:
            augmentation_list.append(
                A.ColorJitter(
                    brightness=jitter_values[0],
                    contrast=jitter_values[1],
                    saturation=jitter_values[2],
                    hue=jitter_values[3],
                    p=p_color_jitter,
                )
            )
        if p_rgb_shift > 0:
            augmentation_list.append(
                A.RGBShift(
                    r_shift_limit=rgb_shift_values[0],
                    g_shift_limit=rgb_shift_values[1],
                    b_shift_limit=rgb_shift_values[2],
                    p=p_color_jitter,
                )
            )
        if p_blur > 0:
            augmentation_list.append(
                A.OneOf(
                    [
                        A.MotionBlur(p=1.0),
                        A.GaussianBlur(p=1.0),
                        A.MedianBlur(p=1.0),
                        A.Blur(p=1.0),
                    ],
                    p=p_blur,
                )
            )
        if p_random_brightness_contrast > 0:
            augmentation_list.append(A.RandomBrightnessContrast(p=p_random_brightness_contrast))
        if p_normalize > 0:
            augmentation_list.append(A.Normalize(mean=ds_mean, std=ds_std, p=p_normalize, max_pixel_value=1.0))

        augmentation_list.append(AP.ToTensorV2())

        self.albu_transform = A.Compose(
            augmentation_list,
            bbox_params=A.BboxParams(format="coco", label_fields=["category_ids"], min_visibility=0.3),
            keypoint_params=A.KeypointParams(
                format="xyas",
                label_fields=["additional_info"],
                angle_in_degrees=True,
                remove_invisible=False,
            ),
        )


class ValTransform(Transform):
    def __init__(
        self,
        max_labels=50,
        image_size=(640, 640),
        p_normalize=0.0,
        ds_mean=(0.485, 0.456, 0.406),
        ds_std=(0.229, 0.224, 0.225),
    ):
        augmentation_list = []

        augmentation_list.append(A.LongestMaxSize(max_size=image_size[0], interpolation=1, p=1.0))
        augmentation_list.append(
            A.PadIfNeeded(
                min_height=image_size[0],
                min_width=image_size[1],
                border_mode=0,
                position="center",
                value=None,
                mask_value=None,
                p=1.0,
            )
        )

        if p_normalize > 0:
            augmentation_list.append(A.Normalize(mean=ds_mean, std=ds_std, p=p_normalize, max_pixel_value=1.0))

        augmentation_list.append(AP.ToTensorV2())

        self.albu_transform = A.Compose(
            augmentation_list,
            bbox_params=A.BboxParams(format="coco", label_fields=["category_ids"]),
            keypoint_params=A.KeypointParams(
                format="xyas",
                label_fields=["additional_info"],
                angle_in_degrees=True,
                remove_invisible=False,
            ),
        )
