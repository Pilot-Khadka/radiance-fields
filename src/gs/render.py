import torch

from gs.camera import Camera
from gs.gaussian import GaussianModel
from gs.project_gaussian_to_2d import calculate_covariance, project_gaussians
from gs.rasterizer import rasterize_gaussians


def render(
    model: GaussianModel,
    camera: Camera,
    background: torch.Tensor,
    render_depth=False,
):
    """
    Render the model from a camera.

    returns:
        image  : (H, W, 3), or (H, W, 4) with depth as last channel
        mean2d : (N, 2) projected centers, for densification gradients
        radii  : (N,) pixel radius, 0 for gaussians not visible
    """
    cov3d = calculate_covariance(model.get_scale(), model.get_rotation())
    mean2d, cov2d, depth, valid = project_gaussians(model.xyz, cov3d, camera)

    features = model.get_color(camera.center)
    if render_depth:
        features = torch.cat([features, depth[:, None]], dim=-1)
        background = torch.cat([background, background.new_zeros(1)])

    image, radii = rasterize_gaussians(
        mean2d[valid],
        cov2d[valid],
        features[valid],
        model.get_opacity()[valid],
        depth[valid],
        image_size=(camera.height, camera.width),
        background=background,
    )

    radii_full = torch.zeros_like(depth)
    radii_full[valid] = radii
    return image, mean2d, radii_full
