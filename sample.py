# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pre-trained DiT.
"""
import torch
import pytorch_lightning as pl
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from download import find_model
from models import DiT_models
import argparse
from PIL import Image
from tqdm import trange # try new and cool stuff!
import numpy as np

def main(args):
    if args.fast_inference:
        tf32 = True
        torch.backends.cudnn.allow_tf32 = bool(tf32)
        torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
        torch.set_float32_matmul_precision('high' if tf32 else 'highest')
        print(f"Fast inference mode is enabled🏎️🏎️🏎️. TF32: {tf32}")
    else:
        print("Fast inference mode is disabled🐢🐢🐢, you may enable it by passing the '--fast-inference' flag!")
    # seed the reproducibility
    pl.seed_everything(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Generating the images by using the device: {device}")

    if args.ckpt is None:
        assert args.model == "DiT-XL/2", "Only DiT-XL/2 models are available for auto-download."
        assert args.image_size in [256, 512]
        assert args.num_classes == 1000

    # Load model:
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
    ).to(device)
    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)
    model.eval()  # important!
    model = torch.compile(model)
    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    # Labels to condition the model with (ImageNet classes):
    if args.start_class is None and args.end_class is not None or args.start_class is not None and args.end_class is None:
        raise ValueError('You can not only provide start-class or end-class. Please provide both or none!')
    if args.start_class and args.end_class:
        class_labels = np.arange(args.start_class, args.end_class, 1).tolist()
        print(f"Generating images for classes {args.start_class} to {args.end_class - 1}  in ImageNet!")
    else:
        class_labels = np.arange(1000).tolist()
        print("Generating images for all 1000 classes in ImageNet!")

    batch_size = args.batch_size
    print("Image Generation Started🔥🔥🔥!")

    for i in trange(0, len(class_labels), batch_size):
        class_index = i
        batch_labels = class_labels[i:i + batch_size]
        generate_samples(batch_labels, model, diffusion, vae, latent_size, device, class_index, args)

        torch.cuda.empty_cache()
    print("Image Generation Completed🎉🎉🎉!")    


def generate_samples(batch_labels, model, diffusion, vae, latent_size, device, class_index, args):
    # Create sampling noise:
    n = len(batch_labels)
    z = torch.randn(n, 4, latent_size, latent_size, device=device)
    y = torch.tensor(batch_labels, device=device)

    # Setup classifier-free guidance:
    z_cfg = torch.cat([z]*2, 0)
    model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)

    # Sample images:
    for i in range(args.images_per_class):
        samples = diffusion.p_sample_loop(
            model.forward_with_cfg, z_cfg.shape, z_cfg, clip_denoised=False, model_kwargs=model_kwargs, progress=False, device=device
        )
        samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
        samples = vae.decode(samples / 0.18215).sample
        samples = torch.clamp(127.5 * samples + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
        
        # Save and display images:
        for j in range(samples.shape[0]):
            index = i + (class_index) * 50
            Image.fromarray(samples[j]).save(f"samples/{index:06d}.png")
        
        # Generate new sampling noise for each iteration
        z_cfg = torch.cat([torch.randn_like(z)]*2, 0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=1.5)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--start-class", type=int, default=None, help="Start class index for Image generation")
    parser.add_argument("--end-class", type=int, default=None, help="End class index for Image generation")
    parser.add_argument("--images-per-class", type=int, default=50, help="Number of images to generate per class")
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--fast-inference", action="store_true", help="Enable fast inference mode.")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    args = parser.parse_args()
    main(args)
