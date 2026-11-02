import torch
import matplotlib.pyplot as plt

from gs.camera import SimpleCamera
from gs.gaussian import GaussianModel
from gs.render import render
from gs.rasterizer import rasterize_gaussians


def show(img, title):
    plt.figure(figsize=(6, 6))
    plt.imshow(img.detach().cpu().clamp(0, 1).numpy())
    plt.title(title)
    plt.axis("off")
    plt.show()


def main():
    # a correct single gaussian should look like a blob
    mean2d = torch.tensor([[64.0, 64.0]])
    cov2d = torch.tensor([[[200.0, 0.0], [0.0, 200.0]]])
    colors = torch.tensor([[1.0, 0.0, 0.0]])
    opacities = torch.tensor([[1.0]])
    depth = torch.tensor([1.0])

    img, _ = rasterize_gaussians(mean2d, cov2d, colors, opacities, depth, (128, 128))
    show(img, "Single Gaussian")

    model = GaussianModel(n_points=1000)
    img, _, radii = render(model, SimpleCamera(), background=torch.zeros(3))
    print("visible:", (radii > 0).sum().item())
    show(img, "Rendered Gaussian Splatting")


if __name__ == "__main__":
    main()
