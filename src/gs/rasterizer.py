"""
Convert each projected gaussian into pixel.

evaluate the 2d gaussian pdf at each pixel
alpha blend the result into a framebuffer, front to back

the image is split into tiles, every gaussian is assigned to the tiles its
visible ellipse touches and sorted by depth inside each tile
"""

from typing import cast

import torch
from torch.utils.checkpoint import checkpoint


def compute_extent(cov2d: torch.Tensor, opacities: torch.Tensor) -> torch.Tensor:
    """
    Pixel half extents (x, y) of the region where alpha >= 1/255.

    alpha = opacity * exp(-0.5 * d), so the Mahalanobis distance d is at most
    2 * log(255 * opacity), faint gaussians cover fewer tiles
    """
    max_dist = (2 * torch.log(255 * opacities[:, 0])).clamp(min=0)
    variance = torch.stack([cov2d[:, 0, 0], cov2d[:, 1, 1]], dim=-1)
    return torch.ceil(torch.sqrt(max_dist[:, None] * variance))


def compute_conic(cov2d: torch.Tensor) -> torch.Tensor:
    """Inverse covariance packed as (a, b, c) for [[a, b], [b, c]]."""
    a, b, c = cov2d[:, 0, 0], cov2d[:, 0, 1], cov2d[:, 1, 1]
    det = (a * c - b * b).clamp(min=1e-12)
    return torch.stack([c / det, -b / det, a / det], dim=-1)


def assign_tiles(mean2d, extent, tile_size, tiles_x, tiles_y):
    """Expand to one entry per (gaussian, tile) overlap."""
    lo = ((mean2d - extent) / tile_size).floor()
    hi = ((mean2d + extent) / tile_size).floor() + 1
    x0 = lo[:, 0].clamp(0, tiles_x).long()
    x1 = hi[:, 0].clamp(0, tiles_x).long()
    y0 = lo[:, 1].clamp(0, tiles_y).long()
    y1 = hi[:, 1].clamp(0, tiles_y).long()
    # gaussians too faint to be seen cover no tile
    visible = (extent > 0).all(dim=-1)
    x1 = torch.where(visible, x1, x0)

    w = x1 - x0
    counts = w * (y1 - y0)
    gauss_ids = torch.repeat_interleave(
        torch.arange(mean2d.shape[0], device=mean2d.device), counts
    )
    starts = torch.cumsum(counts, dim=0) - counts
    local = torch.arange(gauss_ids.shape[0], device=mean2d.device) - starts[gauss_ids]

    tx = x0[gauss_ids] + local % w[gauss_ids]
    ty = y0[gauss_ids] + local // w[gauss_ids]
    return gauss_ids, ty * tiles_x + tx


def render_tiles(pixels, idx, mean2d, conic, colors, opacities, background):
    """
    pixels : (T, P, 2)    pixel coords of every tile
    idx    : (T, K)       gaussian ids per tile front to back, -1 for padding
    output : (T, P, C)
    """
    valid = idx >= 0
    idx = idx.clamp(min=0)

    # offset from center, (T, P, K, 2)
    d = pixels[:, :, None, :] - mean2d[idx][:, None, :, :]
    con = conic[idx][:, None]  # (T, 1, K, 3)

    # Mahalanobis distance
    power = (
        -0.5 * (con[..., 0] * d[..., 0] ** 2 + con[..., 2] * d[..., 1] ** 2)
        - con[..., 1] * d[..., 0] * d[..., 1]
    )
    alpha = (opacities[idx][:, None, :, 0] * torch.exp(power)).clamp(max=0.99)
    alpha = alpha * ((alpha >= 1.0 / 255.0) & (power <= 0) & valid[:, None])

    # transmittance before each gaussian
    T = torch.cumprod(1 - alpha, dim=-1)
    T_prev = torch.cat([torch.ones_like(T[..., :1]), T[..., :-1]], dim=-1)
    weights = alpha * T_prev

    out = torch.einsum("tpk,tkc->tpc", weights, colors[idx])
    return out + T[..., -1:] * background


def plan_chunks(sorted_counts, pixels_per_tile, max_elements):
    """
    Group tiles so tiles * pixels * gaussians stays in budget.

    counts are sorted ascending, so tiles in a chunk need similar padding
    """
    chunks = []
    start = 0
    for i, c in enumerate(sorted_counts):
        k = max(c, 1)
        if i > start and (i - start + 1) * pixels_per_tile * k > max_elements:
            chunks.append((start, i, max(sorted_counts[i - 1], 1)))
            start = i
    chunks.append((start, len(sorted_counts), max(sorted_counts[-1], 1)))
    return chunks


def rasterize_gaussians(
    mean2d,
    cov2d,
    colors,
    opacities,
    depth,
    image_size=(512, 512),
    background=None,
    tile_size=16,
    max_elements=2**22,
):
    """
    mean2d    : (N, 2)           Gaussian center in pixel coords
    cov2d     : (N, 2, 2)        2D covariance from projection
    colors    : (N, C)           any per gaussian feature, rgb, depth, ..
    opacities : (N, 1)
    depth     : (N,)             camera space depth, for sorting

    returns image (H, W, C) and radii (N,), radius is 0 if gaussian is off screen
    """
    device = mean2d.device
    H, W = image_size
    C = colors.shape[-1]
    if background is None:
        background = torch.zeros(C, device=device)

    tiles_x = (W + tile_size - 1) // tile_size
    tiles_y = (H + tile_size - 1) // tile_size
    num_tiles = tiles_x * tiles_y

    with torch.no_grad():
        extent = compute_extent(cov2d, opacities)
        gauss_ids, tile_ids = assign_tiles(mean2d, extent, tile_size, tiles_x, tiles_y)
        radii = extent.max(dim=-1).values

        # sort by depth, then stable sort by tile keeps depth order inside a tile
        order = torch.argsort(depth[gauss_ids], stable=True)
        gauss_ids, tile_ids = gauss_ids[order], tile_ids[order]
        order = torch.argsort(tile_ids, stable=True)
        gauss_ids, tile_ids = gauss_ids[order], tile_ids[order]

        counts = torch.bincount(tile_ids, minlength=num_tiles)
        offsets = torch.cumsum(counts, dim=0) - counts
        # dummy entry so padded lookups stay in range
        gauss_ids = torch.cat([gauss_ids, gauss_ids.new_zeros(1)])
        tile_order = torch.argsort(counts)

        on_screen = torch.zeros_like(radii, dtype=torch.bool)
        on_screen[gauss_ids] = True
        radii = radii * on_screen

    conic = compute_conic(cov2d)

    ys, xs = torch.meshgrid(
        torch.arange(tile_size, device=device),
        torch.arange(tile_size, device=device),
        indexing="ij",
    )
    tile_pixels = torch.stack([xs, ys], dim=-1).reshape(-1, 2).float()  # (P, 2)
    tile_ids_all = torch.arange(num_tiles, device=device)
    tile_origin = torch.stack(
        [(tile_ids_all % tiles_x) * tile_size, (tile_ids_all // tiles_x) * tile_size],
        dim=-1,
    ).float()  # (num_tiles, 2)

    sorted_counts = counts[tile_order].tolist()
    out: list[torch.Tensor] = []
    for t0, t1, k in plan_chunks(sorted_counts, tile_size * tile_size, max_elements):
        tiles = tile_order[t0:t1]
        slot = torch.arange(k, device=device)
        pos = (offsets[tiles][:, None] + slot).clamp(max=gauss_ids.shape[0] - 1)
        idx = torch.where(slot < counts[tiles][:, None], gauss_ids[pos], -1)

        pixels = tile_origin[tiles][:, None, :] + tile_pixels[None]
        args = (pixels, idx, mean2d, conic, colors, opacities, background)
        if torch.is_grad_enabled():
            chunk = checkpoint(render_tiles, *args, use_reentrant=False)
            out.append(cast(torch.Tensor, chunk))
        else:
            out.append(render_tiles(*args))

    image = torch.cat(out)[torch.argsort(tile_order)]
    image = image.reshape(tiles_y, tiles_x, tile_size, tile_size, C)
    image = image.permute(0, 2, 1, 3, 4).reshape(
        tiles_y * tile_size, tiles_x * tile_size, C
    )
    return image[:H, :W], radii
