import torch
from torch import nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors

from gs.sh import eval_sh, rgb_to_sh


def inverse_sigmoid(x: torch.Tensor) -> torch.Tensor:
    return torch.log(x / (1 - x))


class GaussianModel(nn.Module):
    """
    Blob in a space with shape, size and color (tiny 3d cloud)
    Each gaussian has:
        position: (x,y,z)
        covariance: (shape and size, can be stretched and rotated, ellipsoid)
        color: view dependent, spherical harmonics coefficients
        opacity/density
    """

    PARAM_NAMES = ["xyz", "sh_dc", "sh_rest", "log_scale", "rotation", "raw_opacity"]

    def __init__(self, n_points=2000, sh_degree=3, init_range=1.0, xyz=None):
        super().__init__()
        self.sh_degree = sh_degree
        self.active_sh_degree = 0

        if xyz is None:
            xyz = (torch.rand(n_points, 3) * 2 - 1) * init_range
        n_points = xyz.shape[0]
        self.xyz = nn.Parameter(xyz)

        # scale must always be positive
        # init each scale to the mean distance of its 3 nearest neighbours
        points = xyz.numpy()
        nn_dist, _ = NearestNeighbors(n_neighbors=4).fit(points).kneighbors(points)
        mean_dist = torch.tensor(nn_dist[:, 1:], dtype=torch.float32).mean(dim=1)
        self.log_scale = nn.Parameter(
            torch.log(mean_dist.clamp(min=1e-7))[:, None].repeat(1, 3)
        )
        # in quaternion
        self.rotation = nn.Parameter(
            torch.tensor([1.0, 0.0, 0.0, 0.0]).repeat(n_points, 1)
        )
        # initial opacity 0.1
        self.raw_opacity = nn.Parameter(inverse_sigmoid(torch.full((n_points, 1), 0.1)))

        # dc term holds the base color, rest the view dependent part
        self.sh_dc = nn.Parameter(rgb_to_sh(torch.rand(n_points, 1, 3)))
        self.sh_rest = nn.Parameter(torch.zeros(n_points, (sh_degree + 1) ** 2 - 1, 3))

    @property
    def num_points(self):
        return self.xyz.shape[0]

    def get_scale(self):
        return torch.exp(self.log_scale)

    def get_rotation(self):
        # to unit quaternion
        return F.normalize(self.rotation, dim=-1)

    def get_opacity(self):
        return torch.sigmoid(self.raw_opacity)

    def get_color(self, camera_center: torch.Tensor):
        dirs = F.normalize(self.xyz - camera_center, dim=-1)
        sh = torch.cat([self.sh_dc, self.sh_rest], dim=1)
        rgb = eval_sh(self.active_sh_degree, sh, dirs) + 0.5
        return rgb.clamp(min=0.0)

    def increase_sh_degree(self):
        self.active_sh_degree = min(self.active_sh_degree + 1, self.sh_degree)

    def save(self, path):
        params = {name: getattr(self, name).detach().cpu() for name in self.PARAM_NAMES}
        torch.save(
            {
                "sh_degree": self.sh_degree,
                "active_sh_degree": self.active_sh_degree,
                "params": params,
            },
            path,
        )

    @classmethod
    def load(cls, path, device="cpu"):
        ckpt = torch.load(path, map_location=device)
        model = cls(n_points=4, sh_degree=ckpt["sh_degree"])
        for name, value in ckpt["params"].items():
            setattr(model, name, nn.Parameter(value))
        model.active_sh_degree = ckpt["active_sh_degree"]
        return model.to(device)
