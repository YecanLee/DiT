# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pre-trained DiT.
"""
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from download import find_model
from models import DiT_models
import argparse

import numpy as np

def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.ckpt is None:
        assert args.model == "DiT-XL/2", "Only DiT-XL/2 models are available for auto-download."
        assert args.image_size in [256, 512]
        assert args.num_classes == 1000

    # Load model:
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
    ).to(device)
    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)
    model.eval()  # important!
    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    # Labels to condition the model with (ImageNet classes):
    # in case cuda out of memory, at one time please just generate 300 images at most.
    # class_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    class_labels = np.arange(1000)

    batch_size = 1

    for i in range(0, len(class_labels), batch_size):
        batch_labels = class_labels[i:i + batch_size]
        generate_samples(batch_labels, model, diffusion, vae, latent_size, device, args)

        torch.cuda.empty_cache()


def generate_samples(batch_labels, model, diffusion, vae, latent_size, device, args):
    # Create sampling noise:
    n = len(batch_labels)
    z = torch.randn(n, 4, latent_size, latent_size, device=device)
    y = torch.tensor(batch_labels, device=device)

    # Sample 51 images for each class:
    z = z.repeat(51, 1, 1, 1)
    y = y.repeat(51)

    # Setup classifier-free guidance:
    z = torch.cat([z, z], 0)
    y_null = torch.full_like(y, 1000)
    y = torch.cat([y, y_null], 0).long()
    model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)

    # model_kwargs['y'] = model_kwargs['y'].to(torch.long)

    # Sample images:
    samples = diffusion.p_sample_loop(
        model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
    )
    samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
    samples = vae.decode(samples / 0.18215).sample
    # Save all the generated images into "/personal_storage/scout/fid-flaws/data/gen_img_dit"
    
    # Save and display images:
    for i in range(samples.shape[0]):
        class_label = batch_labels[i // 51]
        save_image(samples[i], f"/personal_storage/scout/fid-flaws/data/gen_img_dit/sample_{class_label}_{i % 51}.png", normalize=True, value_range=(-1, 1))
    """
    # Sample images:
    samples = diffusion.p_sample_loop(
        model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
    )
    print(samples.shape, "before the null class removal")
    samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
    samples = vae.decode(samples / 0.18215).sample
    print(samples.shape, "after the null class removal")
    # now all the images are saved in a single image, this is not what we want
    # we want to save each image separately
    # Save and display images:
    for i in range(samples.shape[0]):
        save_image(samples[i], f"/personal_storage/scout/fid-flaws/data/gen_img_dit/sample_{i}.png", normalize=True, value_range=(-1, 1))
    # save_image(samples, "sample.png", nrow=4, normalize=True, value_range=(-1, 1))
    """


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    args = parser.parse_args()
    main(args)
