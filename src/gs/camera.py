from abc import ABC, abstractmethod
import torch


class Camera(ABC):
    @property
    @abstractmethod
    def fx(self) -> float:
        pass

    @property
    @abstractmethod
    def fy(self) -> float:
        pass

    @property
    @abstractmethod
    def cx(self) -> float:
        pass

    @property
    @abstractmethod
    def cy(self) -> float:
        pass

    @property
    @abstractmethod
    def width(self) -> int:
        pass

    @property
    @abstractmethod
    def height(self) -> int:
        pass

    @property
    @abstractmethod
    def R(self) -> torch.Tensor:
        """Rotation matrix (3x3) or (3,3)."""
        pass

    @property
    @abstractmethod
    def t(self) -> torch.Tensor:
        """Translation vector (3,)."""
        pass

    @property
    def center(self) -> torch.Tensor:
        """Camera position in world space (3,)."""
        return -self.R.T @ self.t

    @abstractmethod
    def world_to_camera(self, xyz: torch.Tensor) -> torch.Tensor:
        """Transform points from world space to camera space."""
        pass

    @abstractmethod
    def project(self, xyz: torch.Tensor) -> torch.Tensor:
        """Project 3D points to 2D image coordinates."""
        pass


class SimpleCamera(Camera):
    def __init__(self):
        self._fx = 500.0
        self._fy = 500.0
        self._cx = 256.0
        self._cy = 256.0
        self._width = 512
        self._height = 512

        self._R = torch.eye(3)
        self._t = torch.tensor([0.0, 0.0, 3.0])

    @property
    def fx(self):
        return self._fx

    @property
    def fy(self):
        return self._fy

    @property
    def cx(self):
        return self._cx

    @property
    def cy(self):
        return self._cy

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    @property
    def R(self):
        return self._R

    @property
    def t(self):
        return self._t

    def world_to_camera(self, xyz: torch.Tensor) -> torch.Tensor:
        # X_cam = R @ X + t
        return (self.R @ xyz.T).T + self.t

    def project(self, xyz: torch.Tensor) -> torch.Tensor:
        xyz_cam = self.world_to_camera(xyz)

        x = xyz_cam[:, 0]
        y = xyz_cam[:, 1]
        z = xyz_cam[:, 2]

        u = self.fx * (x / z) + self.cx
        v = self.fy * (y / z) + self.cy

        return torch.stack([u, v], dim=-1)


class PoseCamera(SimpleCamera):
    """Camera from a NeRF synthetic camera-to-world pose (OpenGL convention)."""

    def __init__(self, c2w: torch.Tensor, focal: float, width: int, height: int):
        super().__init__()
        self._fx = focal
        self._fy = focal
        self._cx = width * 0.5
        self._cy = height * 0.5
        self._width = width
        self._height = height

        w2c = torch.linalg.inv(c2w)
        # opengl (y up, z backward) -> opencv (y down, z forward)
        flip = torch.diag(torch.tensor([1.0, -1.0, -1.0], device=c2w.device))
        self._R = flip @ w2c[:3, :3]
        self._t = flip @ w2c[:3, 3]
