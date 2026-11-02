import torch
import torch.nn.functional as F


def gaussian_window(size: int, sigma: float, channels: int, device) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    window = g[:, None] @ g[None, :]
    return window.expand(channels, 1, size, size).contiguous()


def ssim(pred: torch.Tensor, target: torch.Tensor, window_size=11) -> torch.Tensor:
    """
    Structural similarity between two images.

    inputs:
        pred, target: (H, W, C) in [0, 1]
    """
    C1 = 0.01**2
    C2 = 0.03**2

    x = pred.permute(2, 0, 1)[None]
    y = target.permute(2, 0, 1)[None]
    channels = x.shape[1]
    window = gaussian_window(window_size, 1.5, channels, x.device)
    pad = window_size // 2

    def blur(img):
        return F.conv2d(img, window, padding=pad, groups=channels)

    mu_x, mu_y = blur(x), blur(y)
    sigma_x = blur(x * x) - mu_x**2
    sigma_y = blur(y * y) - mu_y**2
    sigma_xy = blur(x * y) - mu_x * mu_y

    ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / (
        (mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2)
    )
    return ssim_map.mean()


def gaussian_splatting_loss(pred, target, lambda_dssim=0.2):
    l1 = (pred - target).abs().mean()
    return (1 - lambda_dssim) * l1 + lambda_dssim * (1 - ssim(pred, target))


def psnr(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return -10.0 * torch.log10(torch.mean((pred - target) ** 2))
