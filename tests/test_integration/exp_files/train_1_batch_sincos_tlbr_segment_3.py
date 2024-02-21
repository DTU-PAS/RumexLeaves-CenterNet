import os
from rumexleaves_centernet.config import Exp as rumex_leaf_exp


class Exp(rumex_leaf_exp):
    def __init__(self):
        super().__init__()
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.gpus = [0]
        self.data_folder = "data/processed/RumexLeaves/iNaturalist"
        self.output_dir = "log/train"
        self.eval_interval = 500
        self.save_interval = 500
        self.print_interval = 1
        self.debug = False

        # Task Weighting
        self.loss_weights = {"hm": 1.0, "off": 1.0, "kp": 10.0, "obb": 10.0, "kphm": 1.0}
        # Model Complexity
        self.arch = "msraresnet_50"
        self.head_complexity = {"hm": 1, "off": 1, "kp": 4, "obb": 4, "kphm": 1}
        self.up_dc = False

        self.target_mode_conf = {
            "do_kp": True,
            "do_kphm": True,
            "do_obb": True,
            "angle_mode": "sincos",  # 360, sincos
            "box_mode": "tlbr",  # wh, tlrb
            "kphm_mode": "segment",  # point, segment
            "cp_i": 3,  # -1 (center point of bounding box), 0, 3, 7
            "normalize": True,
        }

        # Training Configuration
        self.batch_size = 1  # == number of training images, increase when we have more images.
        self.lr = 0.0005
        self.norm_target = True
        self.weight_decay = 5e-5
        self.num_iterations = 5
        self.scheduler = "multistep"
        self.multi_step_milestones = [2500, 4000, 5000]
        self.gamma = 0.5

        # Data configuration
        self.train_split_list = "../../../../../tests/test_integration/exp_files/1_batch_inat.txt"
        self.val_split_list = "../../../../../tests/test_integration/exp_files/1_batch_inat.txt"
        self.input_size = (512, 512)
        self.p_flip = 0.5
        self.p_scale_shift = 0.5
        self.p_rotate = 1.0
        self.p_color_jitter = 1.0
        self.jitter_values = [0.2, 0.2, 0.2, 0.2]
        self.p_random_brightness_contrast = 1.0
        self.ds_mean = (0.43069705, 0.47480908, 0.32581365)  # Format: (R, G, B)
        self.ds_std = (0.43069705, 0.47480908, 0.32581365)  # Format: (R, G, B)

        self.anno_file_id = "annotations_oriented_bb.xml"
