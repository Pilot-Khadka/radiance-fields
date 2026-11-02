from pathlib import Path
import math

import torch
import torchvision.transforms as T
from tqdm import tqdm

from gs.camera import PoseCamera
from gs.densification import Densifier
from gs.gaussian import GaussianModel
from gs.loss import gaussian_splatting_loss, psnr
from gs.render import render
from nerf.data import NeRFDataset
from nerf.utils import get_data_path


RESOLUTION = 400
NUM_POINTS = 100_000
INIT_RANGE = 1.3
SH_DEGREE = 3
ITERATIONS = 30_000
SH_INTERVAL = 1000
LOG_INTERVAL = 100

LR_XYZ_INIT = 1.6e-4
LR_XYZ_FINAL = 1.6e-6
LR_SH = 2.5e-3
LR_OPACITY = 0.025
LR_SCALE = 5e-3
LR_ROTATION = 1e-3

DENSIFY_FROM = 500
DENSIFY_UNTIL = 15_000
DENSIFY_INTERVAL = 100
OPACITY_RESET_INTERVAL = 3000
GRAD_THRESHOLD = 0.0002
MIN_OPACITY = 0.005
MAX_SCREEN_SIZE = 20


def load_split(split, device):
    transform = T.Compose([T.Resize((RESOLUTION, RESOLUTION)), T.ToTensor()])
    dataset = NeRFDataset(
        root_dir=get_data_path() / "lego", split=split, transform=transform
    )
    # focal is given for the original resolution
    focal = dataset.focal * RESOLUTION / dataset.W

    images, cameras = [], []
    for i in range(len(dataset)):
        img, pose = dataset[i]
        images.append(img.permute(1, 2, 0).to(device))
        cameras.append(PoseCamera(pose.to(device), focal, RESOLUTION, RESOLUTION))
    return images, cameras


def scene_extent(cameras):
    centers = torch.stack([cam.center for cam in cameras])
    return 1.1 * (centers - centers.mean(dim=0)).norm(dim=-1).max().item()


def build_optimizer(model: GaussianModel, extent):
    # one group per parameter, densification swaps them by name
    groups = [
        {"name": "xyz", "params": [model.xyz], "lr": LR_XYZ_INIT * extent},
        {"name": "sh_dc", "params": [model.sh_dc], "lr": LR_SH},
        {"name": "sh_rest", "params": [model.sh_rest], "lr": LR_SH / 20},
        {"name": "raw_opacity", "params": [model.raw_opacity], "lr": LR_OPACITY},
        {"name": "log_scale", "params": [model.log_scale], "lr": LR_SCALE},
        {"name": "rotation", "params": [model.rotation], "lr": LR_ROTATION},
    ]
    return torch.optim.Adam(groups, lr=0.0, eps=1e-15)


def xyz_lr(step, extent):
    # log linear decay
    t = step / ITERATIONS
    lr = math.exp(math.log(LR_XYZ_INIT) * (1 - t) + math.log(LR_XYZ_FINAL) * t)
    return lr * extent


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_path = Path("checkpoints")
    save_path.mkdir(exist_ok=True)

    images, cameras = load_split("train", device)
    extent = scene_extent(cameras)
    background = torch.zeros(3, device=device)

    model = GaussianModel(
        n_points=NUM_POINTS, sh_degree=SH_DEGREE, init_range=INIT_RANGE
    ).to(device)
    optimizer = build_optimizer(model, extent)
    densifier = Densifier(model, optimizer, extent)

    progress = tqdm(range(1, ITERATIONS + 1), desc="Training")
    for step in progress:
        optimizer.param_groups[0]["lr"] = xyz_lr(step, extent)
        if step % SH_INTERVAL == 0:
            model.increase_sh_degree()

        img_i = int(torch.randint(0, len(cameras), (1,)))
        image, mean2d, radii = render(model, cameras[img_i], background)
        mean2d.retain_grad()

        loss = gaussian_splatting_loss(image, images[img_i])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        if step < DENSIFY_UNTIL:
            densifier.add_stats(mean2d.grad, radii, (RESOLUTION, RESOLUTION))

            if step > DENSIFY_FROM and step % DENSIFY_INTERVAL == 0:
                # screen size pruning only once opacities have been reset
                max_size = MAX_SCREEN_SIZE if step > OPACITY_RESET_INTERVAL else None
                densifier.densify_and_prune(GRAD_THRESHOLD, MIN_OPACITY, max_size)

            if step % OPACITY_RESET_INTERVAL == 0:
                densifier.reset_opacity()

        optimizer.step()

        if step == 1 or step % LOG_INTERVAL == 0:
            score = psnr(image.detach(), images[img_i])
            progress.set_postfix(
                loss=f"{loss.item():.4f}",
                psnr=f"{score.item():.2f}",
                points=model.num_points,
            )
            # plain line as well, progress bars don't show up in non interactive logs
            elapsed = progress.format_dict["elapsed"]
            eta = elapsed / step * (ITERATIONS - step)
            tqdm.write(
                f"step {step:06d}/{ITERATIONS} | loss {loss.item():.4f} | "
                f"psnr {score.item():.2f} dB | points {model.num_points} | "
                f"elapsed {tqdm.format_interval(elapsed)} | "
                f"eta {tqdm.format_interval(eta)}"
            )

        if step % 5000 == 0 or step == ITERATIONS:
            model.save(save_path / "gs_model.pt")

    print(f"Saved gs_model.pt at {save_path}")


def main():
    train()


if __name__ == "__main__":
    main()
