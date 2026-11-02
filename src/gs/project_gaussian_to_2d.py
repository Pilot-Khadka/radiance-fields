import torch

from gs.camera import SimpleCamera, Camera
from gs.gaussian import GaussianModel
from gs.quaternion import quaternion_to_rot
from gs.plotting import plot_gaussians_depth


def calculate_covariance(
    scale: torch.Tensor,
    rotation: torch.Tensor,
) -> torch.Tensor:
    """
    Given scale and rotation from the gaussian, built the covariance matrix.

        sum = (R.S) (R.S)^T = R.diag(S^2).(R^T)

    inputs:
        scale: (N, 3)
        rotation: (N, 4)
    outputs:
        covariance: (N, 3, 3)
    """
    # (N, 4) -> (N, 3, 3)
    R = quaternion_to_rot(rotation)

    # scale: (N, 3) -> (N, 3, 1)
    sigma = R @ torch.diag_embed(scale**2) @ R.transpose(-1, -2)
    return sigma


def project_gaussians(
    xyz: torch.Tensor,
    cov3d: torch.Tensor,
    camera: Camera,
):
    """Bring the gaussians to camera space first, then project them onto screen."""

    R_cam = camera.R  # (3, 3)
    t_cam = camera.t  # (3,)

    # transfer keypoints from world space to cam space
    p_cam = xyz @ R_cam.T + t_cam  # (N, 3)
    X, Y, Z = p_cam.unbind(-1)

    min_depth = 0.01
    valid = Z > min_depth
    u = camera.fx * X / Z + camera.cx
    v = camera.fy * Y / Z + camera.cy
    mean2d = torch.stack([u, v], dim=-1)  # (N, 2)

    # also rotate world covariance into camera frame.
    # Broadcasting: (3,3) @ (N,3,3) @ (3,3)
    cov_cam = R_cam @ cov3d @ R_cam.T  # (N, 3, 3)

    # the jacobian is only a good approximation near the view frustum
    lim_x = 1.3 * 0.5 * camera.width / camera.fx
    lim_y = 1.3 * 0.5 * camera.height / camera.fy
    X = (X / Z).clamp(-lim_x, lim_x) * Z
    Y = (Y / Z).clamp(-lim_y, lim_y) * Z

    zeros = torch.zeros_like(Z)
    J = torch.stack(
        [
            camera.fx / Z,
            zeros,
            -camera.fx * X / Z**2,
            zeros,
            camera.fy / Z,
            -camera.fy * Y / Z**2,
        ],
        dim=-1,
    ).reshape(-1, 2, 3)  # (N, 2, 3)

    cov2d = J @ cov_cam @ J.transpose(-1, -2)  # (N, 2, 2)
    # low pass filter, every gaussian covers at least about one pixel
    cov2d = cov2d + 0.3 * torch.eye(2, device=cov2d.device)

    return mean2d, cov2d, Z, valid


def main():
    N = 1000
    min_depth = 1.0
    max_depth = 4.0
    model = GaussianModel(n_points=N)

    xyz = model.xyz
    # scale shape: (N, 3)
    scale = model.get_scale()
    # rotation shape: (N, 4)
    rotation = model.get_rotation()

    cov3d = calculate_covariance(scale=scale, rotation=rotation)
    camera = SimpleCamera()

    mean2d, cov2d, depth, valid = project_gaussians(xyz, cov3d, camera)
    plot_gaussians_depth(mean2d, cov2d, depth, valid)

    print("2D means:", mean2d.shape)  # (N, 2)
    print("2D cov:", cov2d.shape)  # (N, 2, 2)
    print("Depth:", depth.shape)  # (N,)
    print("Valid:", valid.shape)  # (N,)

    eigvals = torch.linalg.eigvals(cov2d[valid])
    print("Cov2D eigenvalues (real part):", eigvals.real.mean())

    near = cov2d[(depth < min_depth) & valid]
    far = cov2d[(depth > max_depth) & valid]

    print(near.mean(), far.mean())

    i = 0
    print("cov2d:", cov2d[i])
    eigvals, eigvecs = torch.linalg.eigh(cov2d[i])
    print("major axis direction:", eigvecs[:, 1])

    print("min depth:", depth.min().item())
    print("max depth:", depth.max().item())


if __name__ == "__main__":
    main()
