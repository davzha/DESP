import torch
import torch.nn as nn


def langevin_sample(energy_fn, x, n_points, d_y, T, S, eps, device="cuda"):
    """Also used for prediction, by setting S < T.
    """
    batch_size = x.size(0)
    y_t = torch.zeros(batch_size, n_points, d_y, device=device)

    energies = []
    gradnorm = torch.zeros(1, device=device, requires_grad=False)

    for i in range(T):
        with torch.enable_grad():
            y_t.requires_grad = True
            if S is None or i < S:
                # This is slightly different from the paper, 
                # but offers small performance boosts in some cases.
                # The change amounts to dropping the noise on the last sampling step.
                y_t.data += eps * torch.randn_like(y_t)

            e = energy_fn(y_t, x)

            grads, = torch.autograd.grad(
                inputs=y_t,
                outputs=e.sum(),
                only_inputs=True,
                retain_graph=True,
                allow_unused=False
            )

            gradnorm += grads.contiguous().view(batch_size, -1).norm(dim=1).mean(0)
            y_t.data -= grads

        energies.append(e.detach())

    energies = torch.stack(energies)

    return y_t.detach(), dict(energies=energies, gradnorm=gradnorm / T)


class ClippedGradientDescent(nn.Module):
    def __init__(self, energy_fn, n_points, d_y, T, eps, norm_type, max_norm):
        super().__init__()
        self.energy_fn = energy_fn
        self.n_points = n_points
        self.d_y = d_y
        self.eps = eps
        self.T = T
        self.norm_type = norm_type
        self.max_norm = max_norm

        self.y_init = nn.Parameter(torch.Tensor(1, n_points, d_y))
        nn.init.uniform_(self.y_init)

    def clip_gradient(self, grads):
        if self.norm_type is None:
            return grads

        if self.norm_type == 0.:
            return grads.clamp(-self.max_norm, self.max_norm)
        
        grad_norm = grads.norm(2., dim=(1,2), keepdim=True)
        clip_coef = self.max_norm / (grad_norm.detach() + 1e-6)
        clip_coef = clip_coef.clamp(0., 1.)
        grads = grads * clip_coef
        return grads

    def forward(self, x):
        batch_size = x.size(0)
        y_t = self.y_init.expand(batch_size, -1, -1) 

        energies = []
        ys = [y_t]
        gradnorms = torch.zeros(1, device=y_t.device)

        for _ in range(self.T):
            with torch.enable_grad():
                if not self.training:
                    y_t.requires_grad = True

                e = self.energy_fn(y_t, x).sum(0)

                grads, = torch.autograd.grad(
                    inputs=y_t,
                    outputs=e,
                    only_inputs=True,
                    create_graph=True,
                    allow_unused=False
                )

            grads = self.clip_gradient(grads)
            y_t = y_t - self.eps * grads

            ys.append(y_t)
            energies.append(e)
            gradnorms += grads.contiguous().detach().view(batch_size, -1).norm(1).mean()

        return ys, dict(energies=energies, gradnorm=gradnorms / self.T)

