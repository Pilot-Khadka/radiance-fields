from pathlib import Path
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from nerf.model import PositionalEncoding, MLP
from nerf.data import NeRFDataset
from nerf.utils import get_data_path


NEAR = 2.0
FAR = 6.0
NUM_SAMPLES = 64
BATCH_SIZE = 512
NUM_EPOCHS = 20
LR = 5e-4


def setup_ddp(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_ddp():
    dist.destroy_process_group()


def volume_render(rgb, sigma, z_vals):
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat([dists, torch.full_like(dists[..., :1], 1e10)], dim=-1)

    alpha = 1.0 - torch.exp(-sigma[..., 0] * dists)

    transmittance = torch.cumprod(
        torch.cat([torch.ones_like(alpha[..., :1]), 1.0 - alpha + 1e-10], dim=-1),
        dim=-1,
    )[..., :-1]

    weights = transmittance * alpha

    rgb_map = (weights[..., None] * rgb).sum(dim=-2)
    depth_map = (weights * z_vals).sum(dim=-1)
    acc_map = weights.sum(dim=-1)

    return rgb_map, depth_map, acc_map


def get_rays(H, W, focal, pose):
    i, j = torch.meshgrid(
        torch.arange(W, dtype=torch.float32, device=pose.device),
        torch.arange(H, dtype=torch.float32, device=pose.device),
        indexing="xy",
    )
    # convert pixel coordinates to camera centered coordinates
    # origin is at image center, not the top left corner
    # /focal converts pixel units to cam rays
    dirs = torch.stack(
        [(i - W * 0.5) / focal, -(j - H * 0.5) / focal, -torch.ones_like(i)],
        dim=-1,
    )

    # convert ray direction in cam space to world space
    rays_d = (dirs[..., None, :] * pose[:3, :3]).sum(-1)
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    rays_o = pose[:3, 3].expand(rays_d.shape)
    return rays_o, rays_d


def sample_points(
    rays_o,
    rays_d,
    near,
    far,
    num_samples,
    perturb=True,
):
    # creates a sample index
    # z(t) = (1-t) * neat + t * far
    # produces evely spaced depth samples
    t_vals = torch.linspace(0.0, 1.0, num_samples, device=rays_o.device)
    z_vals = near * (1 - t_vals) + far * t_vals
    z_vals = z_vals.expand(*rays_o.shape[:-1], num_samples)

    if perturb:
        # stratified monte carlo sampling
        # pick random depth between bounds
        mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], dim=-1)
        lower = torch.cat([z_vals[..., :1], mids], dim=-1)
        # adds controlled noise
        z_vals = lower + (upper - lower) * torch.rand_like(z_vals)

    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
    return pts, z_vals


def build_ray_batch(dataset, device):
    H, W, focal = dataset.H, dataset.W, dataset.focal
    all_rays_o, all_rays_d, all_pixels = [], [], []

    for i in range(len(dataset)):
        img, pose = dataset[i]
        rays_o, rays_d = get_rays(H, W, focal, pose.to(device))
        all_rays_o.append(rays_o.reshape(-1, 3))
        all_rays_d.append(rays_d.reshape(-1, 3))
        all_pixels.append(img.permute(1, 2, 0).reshape(-1, 3).to(device))

    rays_o = torch.cat(all_rays_o)
    rays_d = torch.cat(all_rays_d)
    pixels = torch.cat(all_pixels)

    perm = torch.randperm(rays_o.shape[0], device=device)
    return rays_o[perm], rays_d[perm], pixels[perm]


def train(rank, world_size):
    is_distributed = world_size > 1

    if is_distributed:
        setup_ddp(rank, world_size)

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    save_path = Path("checkpoints")
    save_path.mkdir(exist_ok=True)

    data_root = get_data_path() / "lego"
    dataset = NeRFDataset(root_dir=data_root, split="train")

    pos_enc_freq = 10
    dir_enc_freq = 4
    pos_enc = PositionalEncoding(num_freqs=pos_enc_freq, include_input=True)
    dir_enc = PositionalEncoding(num_freqs=dir_enc_freq, include_input=True)

    pos_dim = 3 * (1 + 2 * pos_enc_freq)
    dir_dim = 3 * (1 + 2 * dir_enc_freq)

    model = MLP(pos_dim=pos_dim, dir_dim=dir_dim, hidden_dim=256).to(device)

    if is_distributed:
        model = DDP(model, device_ids=[rank])

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    # each rank process a portion of total_steps, which are combined
    total_steps = NUM_EPOCHS * (dataset.H * dataset.W // BATCH_SIZE)
    steps_per_rank = total_steps // world_size

    for step in range(steps_per_rank):
        img_i = torch.randint(0, len(dataset), (1,)).item()
        img, pose = dataset[img_i]
        img = img.to(device)
        pose = pose.to(device)

        rays_o, rays_d = get_rays(dataset.H, dataset.W, dataset.focal, pose)
        coords = torch.randint(0, dataset.H * dataset.W, (BATCH_SIZE,), device=device)
        batch_o = rays_o.reshape(-1, 3)[coords]
        batch_d = rays_d.reshape(-1, 3)[coords]
        batch_px = img.permute(1, 2, 0).reshape(-1, 3)[coords]

        pts, z_vals = sample_points(
            batch_o, batch_d, NEAR, FAR, NUM_SAMPLES, perturb=True
        )
        B, S = pts.shape[:2]

        pts_enc = pos_enc.encode(pts.reshape(-1, 3))
        dirs_enc = dir_enc.encode(batch_d[:, None].expand(B, S, 3).reshape(-1, 3))

        rgb_raw, sigma = model(pts_enc, dirs_enc)
        rgb_map, _, _ = volume_render(
            rgb=rgb_raw.reshape(B, S, 3),
            sigma=sigma.reshape(B, S, 1),
            z_vals=z_vals,
        )

        loss = torch.mean((rgb_map - batch_px) ** 2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if rank == 0 and step > 1 and step % 500 == 0:
            psnr = -10.0 * torch.log10(loss)
            global_step = step * world_size
            print(
                f"step {global_step:06d}/{total_steps} | loss {loss.item():.4f} | psnr {psnr.item():.2f} dB"
            )

    if rank == 0:
        state_dict = model.module.state_dict() if is_distributed else model.state_dict()
        torch.save(state_dict, save_path / "nerf_model.pth")
        print(f"Saved nerf_model.pth at {save_path}")

    if is_distributed:
        cleanup_ddp()


def main():
    world_size = torch.cuda.device_count()

    if world_size > 1:
        print(f"Launching DDP training on {world_size} GPUs")
        mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)
    else:
        print("Single GPU (or CPU) training")
        train(rank=0, world_size=1)


if __name__ == "__main__":
    main()
