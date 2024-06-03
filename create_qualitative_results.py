import sys
import os
import shutil
sys.path.insert(0, '/mnt/users/ericryanchan/repos')

import dit_k as DiT
import torch
from torchvision.utils import save_image
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
# from download import find_model
# from models import DiT_XL_2
# from mamba_models import Mamba_M
# from models_improved_transformer import DiT_plus
from experiment_helpers import experiments


seed = 12


from PIL import Image
from IPython.display import display
torch.set_grad_enabled(False)
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    print("GPU not found. Using CPU instead.")



def sample_images(model, output_path="sample.png", seed=0):
    # Set user inputs:
    torch.manual_seed(seed)
    num_sampling_steps = 250 #@param {type:"slider", min:0, max:1000, step:1}
    cfg_scale = 1 #@param {type:"slider", min:1, max:10, step:0.1}
    # class_labels = 207, 360, 387, 974, 88, 979, 417, 279 #@param {type:"raw"}
    class_labels = [0] * 16
    samples_per_row = 8 #@param {type:"number"}

    # Create diffusion object:
    diffusion = create_diffusion(str(num_sampling_steps))

    # Create sampling noise:
    n = len(class_labels)
    z = torch.randn(n, 4, latent_size, latent_size, device=device)
    y = torch.tensor(class_labels, device=device)

    # Setup classifier-free guidance:
    z = torch.cat([z, z], 0)
    y_null = torch.tensor([1000] * n, device=device)
    y = torch.cat([y, y_null], 0)
    model_kwargs = dict(y=y, cfg_scale=cfg_scale)

    # Sample images:
    samples = diffusion.p_sample_loop(
        model.forward_with_cfg, z.shape, z, clip_denoised=False, 
        model_kwargs=model_kwargs, progress=True, device=device
    )
    samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
    samples = vae.decode(samples / 0.18215).sample

    # Save and display images:
    save_image(samples, output_path, nrow=int(samples_per_row), 
            normalize=True, value_range=(-1, 1))



image_size = 256
vae_model = "stabilityai/sd-vae-ft-ema"
latent_size = int(image_size) // 8
vae = AutoencoderKL.from_pretrained(vae_model).to(device)

out_dir = "qualitative_results"
# if os.path.exists(out_dir):
#     shutil.rmtree(out_dir)
os.makedirs(out_dir)

with torch.no_grad():
    for exp in experiments:
        model = exp.instantiate().to(device)

        sample_images(model, output_path=os.path.join(out_dir, exp.tag + ".png"), seed=seed)
