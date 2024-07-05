import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__name__), '../../../')))

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from DiT.diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from DiT.download import find_model
from DiT.models import DiT_models
import argparse
import numpy as np

def main(args):
    total_samples = 50000
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu" 

    if args.ckpt is None:
        assert args.model == "DiT-XL/2", "Only DiT-XL/2 models are available for auto-download."
        assert args.image_size in [256, 512]
        assert args.num_classes == 1000

    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
    ).to(device)
    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)
    model.eval()
    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    class_labels = np.arange(1000)
    for class_label in class_labels:
        batch_labels = torch.tensor([class_label] * args.images_per_class).to(device)
        with torch.inference_mode():
            generate_samples(batch_labels, model, diffusion, vae, latent_size, device, args, total_samples)
        total_samples += args.images_per_class
        torch.cuda.empty_cache()

def generate_samples(batch_labels, model, diffusion, vae, latent_size, device, args, total):
    n = len(batch_labels)
    z = torch.randn(n, 4, latent_size, latent_size, device=device)
    y = batch_labels.clone().detach()

    z = torch.cat([z, z], 0)
    y_null = torch.full_like(y, 1000)
    y = torch.cat([y, y_null], 0).long()
    model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)

    samples = diffusion.p_sample_loop(
        model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
    )
    samples, _ = samples.chunk(2, dim=0)
    samples = vae.decode(samples / 0.18215).sample
    
    for i in range(samples.shape[0]):
        save_image(samples[i], f"../../data/DiT/dit_real_distrbution/{total + i:06d}.png", normalize=True, value_range=(-1, 1))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=1.5) 
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prefix", type=str, default="fid-flaws/data/DiT/dit_real_distrbution/")
    parser.add_argument("--images-per-class", type=int, default=30)
    parser.add_argument("--num-samples", type=int, default=30000)
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    args = parser.parse_args()
    main(args)