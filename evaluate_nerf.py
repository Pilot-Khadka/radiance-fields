import torch
from PIL import Image
from pathlib import Path


import numpy as np
from tqdm import tqdm
import torchvision.transforms as T


from nerf.data import NeRFDataset
from nerf.utils import get_data_path
from nerf.model import PositionalEncoding, MLP
from train import get_rays, sample_points, volume_render

NEAR = 2.0
FAR = 6.0
NUM_SAMPLES = 32
CHUNK = 4096


@torch.no_grad()
def render_image(
    model,
    pos_enc,
    dir_enc,
    pose,
    H,
    W,
    focal,
    near,
    far,
    num_samples,
    device: torch.device,
    chunk=2048,
):
    """
    Renders one full image by splitting rays into chunks to stay within memory.

    perturb=False so the output is deterministic at eval/render time.
    """
    rays_o, rays_d = get_rays(H, W, focal, pose.to(device))
    rays_o = rays_o.reshape(-1, 3)
    rays_d = rays_d.reshape(-1, 3)

    rgb_chunks, depth_chunks = [], []

    for i in tqdm(range(0, rays_o.shape[0], chunk), desc="Rendering rays", leave=False):
        batch_o = rays_o[i : i + chunk]
        batch_d = rays_d[i : i + chunk]

        pts, z_vals = sample_points(
            batch_o, batch_d, near, far, num_samples, perturb=False
        )
        B, S = pts.shape[:2]

        pts_enc = pos_enc.encode(pts.reshape(-1, 3))
        dirs_enc = dir_enc.encode(batch_d[:, None].expand(B, S, 3).reshape(-1, 3))

        rgb_raw, sigma = model(pts_enc, dirs_enc)
        rgb_map, depth_map, _ = volume_render(
            rgb_raw.reshape(B, S, 3),
            sigma.reshape(B, S, 1),
            z_vals,
        )

        rgb_chunks.append(rgb_map.cpu())
        depth_chunks.append(depth_map.cpu())

    rgb = torch.cat(rgb_chunks).reshape(H, W, 3)
    depth = torch.cat(depth_chunks).reshape(H, W)
    return rgb, depth


def psnr(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return -10.0 * torch.log10(torch.mean((pred - target) ** 2))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path("eval_results")
    out_dir.mkdir(exist_ok=True)

    data_root = get_data_path() / "lego"
    dataset = NeRFDataset(
        root_dir=data_root,
        split="test",
        transform=T.Compose([T.ToTensor()]),
    )

    pos_enc = PositionalEncoding(num_freqs=10, include_input=True)
    dir_enc = PositionalEncoding(num_freqs=4, include_input=True)
    pos_dim = 3 * (1 + 2 * 10)
    dir_dim = 3 * (1 + 2 * 4)

    model = MLP(pos_dim=pos_dim, dir_dim=dir_dim, hidden_dim=256).to(device)
    model.load_state_dict(torch.load("checkpoints/nerf_model.pth", map_location=device))
    model.eval()

    psnr_scores = []

    for i in tqdm(range(len(dataset)), desc="Rendering images"):
        gt_img, pose = dataset[i]

        rgb, depth = render_image(
            model,
            pos_enc,
            dir_enc,
            pose,
            dataset.H,
            dataset.W,
            dataset.focal,
            NEAR,
            FAR,
            NUM_SAMPLES,
            device,
            CHUNK,
        )

        gt = gt_img.permute(1, 2, 0)
        score = psnr(rgb, gt)
        psnr_scores.append(score.item())

        # Left: ground truth | Right: rendered
        comparison = torch.cat([gt, rgb], dim=1).clamp(0, 1)
        Image.fromarray((comparison.numpy() * 255).astype(np.uint8)).save(
            out_dir / f"test_{i:03d}.png"
        )

        # Depth map saved separately, normalized to [0, 1]
        depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        Image.fromarray((depth_norm.numpy() * 255).astype(np.uint8)).save(
            out_dir / f"depth_{i:03d}.png"
        )

        print(f"[{i + 1:03d}/{len(dataset)}] PSNR: {score.item():.2f} dB")

    mean_psnr = np.mean(psnr_scores)
    std_psnr = np.std(psnr_scores)
    print(f"\nMean PSNR: {mean_psnr:.2f} ± {std_psnr:.2f} dB")
    print(f"Results saved to {out_dir}/")


if __name__ == "__main__":
    main()
