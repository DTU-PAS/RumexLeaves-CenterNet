import argparse
import torch

from rumexleaves_centernet.config import get_exp
from rumexleaves_centernet.utils import load_ckpt


class TestSampleEvaluation:
    ckpt_path = "models/final_model.pth"

    def perform_evluation(self, exp_file):
        exp = get_exp(exp_file)
        model = exp.get_model()
        device = exp.get_device()
        ckpt = torch.load(self.ckpt_path, map_location=device)["model"]
        model = load_ckpt(model, ckpt)
        model.to(device)

        args = argparse.Namespace()
        args.experiment_name = exp.exp_name
        evaluator = exp.get_evaluator(args)
        (performance, loss_meter), bb_detections, kp_detections = exp.eval(model, evaluator)

        print(performance)

        return performance

    def test_final_model_on_inat(self):
        exp_file = "tests/test_integration/exp_files/eval_inat.py"

        performance = self.perform_evluation(exp_file)

        assert performance["mAP"]["ap50_95"] == 0.7088059014771247
        assert performance["OKS"]["kp_all"]["ap50_95"] == 0.44621290596435714

    def test_final_model_on_roborumex(self):
        exp_file = "tests/test_integration/exp_files/eval_roborumex.py"
        performance = self.perform_evluation(exp_file)

        assert performance["mAP"]["ap50_95"] == 0.6706554197898036
        assert performance["OKS"]["kp_all"]["ap50_95"] == 0.29022879751186004
