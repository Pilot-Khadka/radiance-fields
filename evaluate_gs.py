import torch
from PIL import Image
from pathlib import Path


import numpy as np
from tqdm import tqdm


from gs.gaussian import GaussianModel
from gs.loss import psnr
from gs.render import render
from train_gs import load_split


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path("eval_results_gs")
    out_dir.mkdir(exist_ok=True)

    images, cameras = load_split("test", device)
    background = torch.zeros(3, device=device)

    model = GaussianModel.load("checkpoints/gs_model.pt", device=device)
    print(f"Loaded {model.num_points} gaussians")

    psnr_scores = []

    for i in tqdm(range(len(cameras)), desc="Rendering images"):
        with torch.no_grad():
            out, _, _ = render(model, cameras[i], background, render_depth=True)
        rgb = out[..., :3].clamp(0, 1).cpu()
        depth = out[..., 3].cpu()

        gt = images[i].cpu()
        score = psnr(rgb, gt)
        psnr_scores.append(score.item())

        # Left: ground truth | Right: rendered
        comparison = torch.cat([gt, rgb], dim=1)
        Image.fromarray((comparison.numpy() * 255).astype(np.uint8)).save(
            out_dir / f"test_{i:03d}.png"
        )

        # Depth map saved separately, normalized to [0, 1]
        depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        Image.fromarray((depth_norm.numpy() * 255).astype(np.uint8)).save(
            out_dir / f"depth_{i:03d}.png"
        )

        print(f"[{i + 1:03d}/{len(cameras)}] PSNR: {score.item():.2f} dB")

    mean_psnr = np.mean(psnr_scores)
    std_psnr = np.std(psnr_scores)
    print(f"\nMean PSNR: {mean_psnr:.2f} ± {std_psnr:.2f} dB")
    print(f"Results saved to {out_dir}/")


if __name__ == "__main__":
    main()
