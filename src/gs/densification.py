import torch
from torch import nn

from gs.gaussian import GaussianModel, inverse_sigmoid
from gs.quaternion import quaternion_to_rot


def repeat_rows(x: torch.Tensor, n: int) -> torch.Tensor:
    return x.repeat(n, *[1] * (x.dim() - 1))


class Densifier:
    """
    Adaptive density control.

    gaussians with a large view space gradient sit in under reconstructed regions:
        small ones are cloned
        large ones are split into two smaller ones
    almost transparent or oversized gaussians are pruned

    optimizer param groups must be named after the model parameters
    """

    def __init__(self, model: GaussianModel, optimizer, extent, percent_dense=0.01):
        self.model = model
        self.optimizer = optimizer
        self.extent = extent
        self.percent_dense = percent_dense

        n = model.num_points
        device = model.xyz.device
        self.grad_accum = torch.zeros(n, device=device)
        self.denom = torch.zeros(n, device=device)
        self.max_radii = torch.zeros(n, device=device)

    @torch.no_grad()
    def add_stats(self, mean2d_grad, radii, image_size):
        H, W = image_size
        visible = radii > 0
        # gradient in ndc space, keeps the threshold independent of resolution
        ndc_scale = torch.tensor([W * 0.5, H * 0.5], device=mean2d_grad.device)
        grad_norm = (mean2d_grad[visible] * ndc_scale).norm(dim=-1)

        self.grad_accum[visible] += grad_norm
        self.denom[visible] += 1
        self.max_radii[visible] = torch.maximum(self.max_radii[visible], radii[visible])

    def _replace(self, param_fn, state_fn, names=None):
        """Swap parameters for param_fn(name, param), keeping adam state in sync."""
        for group in self.optimizer.param_groups:
            name = group["name"]
            if names is not None and name not in names:
                continue

            old = group["params"][0]
            new = nn.Parameter(param_fn(name, old.detach()))
            state = self.optimizer.state.pop(old, None)
            if state:
                state["exp_avg"] = state_fn(state["exp_avg"])
                state["exp_avg_sq"] = state_fn(state["exp_avg_sq"])
                self.optimizer.state[new] = state

            group["params"][0] = new
            setattr(self.model, name, new)

    def _append(self, new_params: dict):
        n = new_params["xyz"].shape[0]
        self._replace(
            lambda name, p: torch.cat([p, new_params[name]]),
            lambda s: torch.cat([s, s.new_zeros((n, *s.shape[1:]))]),
        )
        self.max_radii = torch.cat([self.max_radii, self.max_radii.new_zeros(n)])

    def _keep(self, mask: torch.Tensor):
        self._replace(lambda name, p: p[mask], lambda s: s[mask])
        self.max_radii = self.max_radii[mask]

    def _clone(self, grads, grad_threshold):
        max_scale = self.model.get_scale().max(dim=1).values
        mask = (grads >= grad_threshold) & (
            max_scale <= self.percent_dense * self.extent
        )
        self._append(
            {
                name: getattr(self.model, name).detach()[mask]
                for name in GaussianModel.PARAM_NAMES
            }
        )

    def _split(self, grads, grad_threshold, n_split=2):
        n = self.model.num_points
        # clone already appended new gaussians, they are not split
        padded = torch.zeros(n, device=grads.device)
        padded[: grads.shape[0]] = grads

        scale = self.model.get_scale()
        mask = (padded >= grad_threshold) & (
            scale.max(dim=1).values > self.percent_dense * self.extent
        )

        # sample new centers from the gaussian itself
        stds = repeat_rows(scale[mask], n_split)
        samples = torch.randn_like(stds) * stds
        R = repeat_rows(quaternion_to_rot(self.model.get_rotation()[mask]), n_split)
        xyz = repeat_rows(self.model.xyz[mask], n_split)

        new_params = {
            name: repeat_rows(getattr(self.model, name).detach()[mask], n_split)
            for name in GaussianModel.PARAM_NAMES
        }
        new_params["xyz"] = (R @ samples[..., None])[..., 0] + xyz
        new_params["log_scale"] = torch.log(stds / (0.8 * n_split))
        self._append(new_params)

        # remove the originals that were split
        n_new = int(mask.sum()) * n_split
        self._keep(~torch.cat([mask, mask.new_zeros(n_new)]))

    @torch.no_grad()
    def densify_and_prune(self, grad_threshold, min_opacity, max_screen_size=None):
        grads = self.grad_accum / self.denom.clamp(min=1)
        self._clone(grads, grad_threshold)
        self._split(grads, grad_threshold)

        prune = self.model.get_opacity()[:, 0] < min_opacity
        if max_screen_size is not None:
            prune |= self.max_radii > max_screen_size
            prune |= self.model.get_scale().max(dim=1).values > 0.1 * self.extent
        self._keep(~prune)

        n = self.model.num_points
        self.grad_accum = self.grad_accum.new_zeros(n)
        self.denom = self.denom.new_zeros(n)
        self.max_radii = self.max_radii.new_zeros(n)

    @torch.no_grad()
    def reset_opacity(self):
        opacity = self.model.get_opacity().clamp(max=0.01)
        self._replace(
            lambda name, p: inverse_sigmoid(opacity),
            torch.zeros_like,
            names=["raw_opacity"],
        )
