import os
import torch

class Experiment:
    def __init__(
            self,
            tag,
            exp_path,
            model_class,
            ):
        
        self.tb_log_path = [os.path.join(exp_path, f) for f in os.listdir(exp_path) if "events.out.tfevents" in f][0]
        self.checkpoint_path = sorted([os.path.join(exp_path, "checkpoints", f) for f in os.listdir(os.path.join(exp_path, "checkpoints"))], key=lambda x: int(x.split("/")[-1].split(".")[0]))[-1]
        self.model_class = model_class
        self.tag = tag

    def instantiate(self):
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
        if "ema" in checkpoint:  # supports checkpoints from train.py
            checkpoint = checkpoint["ema"]
        model = self.model_class(input_size=256//8)
        model.load_state_dict(checkpoint)
        model.eval()
        return model

    def get_training_loss(self):
        from tensorboard.backend.event_processing import event_accumulator
        ea = event_accumulator.EventAccumulator(self.tb_log_path)
        ea.Reload()
        scalars = ea.scalars.Items("train/loss")
        steps = [scalar.step for scalar in scalars]
        values = [scalar.value for scalar in scalars]
        return steps, values

    def get_validation_loss(self):
        from tensorboard.backend.event_processing import event_accumulator
        ea = event_accumulator.EventAccumulator(self.tb_log_path)
        ea.Reload()
        scalars = ea.scalars.Items("val/loss")
        steps = [scalar.step for scalar in scalars]
        values = [scalar.value for scalar in scalars]
        return steps, values
        

from models import DiT_XL_2, DiT_XL_2_Swin7, DiT_XL_2_Linformer256
from mamba_models import Mamba_M
from hybrid_models import HybridDiT_XL_2_Slow, HybridDiT_XL_2_Fast_Comp, HybridDiT_XL_2_Fast_Comp_ReLU
from models_improved_transformer import DiT_plus_XL_2


path_prefix = "/mnt/users/ericryanchan/repos/dit_k"

experiments = [
    Experiment(
        "baseline",
        exp_path=path_prefix + "results/000-DiT-XL-2",
        model_class=DiT_XL_2,
    ),
    Experiment(
        "mamba",
        exp_path=path_prefix + "results-mamba/003-Mamba-M-2",
        model_class=Mamba_M,
    ),
    Experiment(
        "hybrid",
        exp_path=path_prefix + "results-hybrid/005-HybridDiT-XL-2-Slow",
        model_class=HybridDiT_XL_2_Slow,
    ),
    Experiment(
        "hybrid downsample",
        exp_path=path_prefix + "results-hybrid/006-HybridDiT-XL-2-Fast-Comp",
        model_class=HybridDiT_XL_2_Fast_Comp,
    ),
    Experiment(
        "hybrid downsample ReLU",
        exp_path=path_prefix + "results-hybrid/005-HybridDiT-XL-2-Fast-Comp-ReLU",
        model_class=HybridDiT_XL_2_Fast_Comp_ReLU,
    ),
    Experiment(
        "weight decay",
        exp_path=path_prefix + "results-wd-1e-2/000-HybridDiT-XL-2",
        model_class=DiT_XL_2,
    ),
    Experiment(
        "swin",
        exp_path=path_prefix + "results-swin/000-DiT-XL-2-Swin7",
        model_class=DiT_XL_2_Swin7,
    ),
    Experiment(
        "linformer",
        exp_path=path_prefix + "results-linformer/001-DiT_XL-2-Linformer256",
        model_class=DiT_XL_2_Linformer256,
    ),
    Experiment(
        "transformer++",
        exp_path=path_prefix + "results-DiT++/000-DiT++-XL-2",
        model_class=DiT_plus_XL_2,
    ),

]