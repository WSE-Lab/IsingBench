import torch

from .base_solver import *
import numpy as np


class BaseIsingSolver(BaseSolver, ABC):
    """
    Base class for solvers that operate directly on the Ising formulation.

    Extends :class:`BaseSolver` by accepting J and h at construction time
    and providing a batched energy evaluation utility.

    Args:
        J:      ``(N, N)`` symmetric coupling matrix with zero diagonal.
        h:      ``(N,)`` external-field bias vector.
        device: PyTorch device string, e.g. ``"cpu"`` or ``"cuda"``.
    """
    def __init__(self, J: np.ndarray, h: np.ndarray, device: str = "cpu"):
        super().__init__(device)
        self.N = J.shape[0]
        self.device = device
        self.J = torch.from_numpy(J).float().to(device)
        self.h = torch.from_numpy(h).float().to(device)

    def calc_ising_energy(self, spins: torch.Tensor) -> torch.Tensor:
        """
        Compute the Ising energy for a batch of spin configurations.

        The energy for a single configuration **s** is defined as:

        .. math::

            E(\\mathbf{s}) = -\\frac{1}{2} \\mathbf{s}^\\top J \\mathbf{s} - \\mathbf{h}^\\top \\mathbf{s}

        Args:
            spins: ``(batch_size, N)`` tensor of spin configurations
                   with values in ``{-1, +1}``.

        Returns:
            ``(batch_size,)`` tensor of energy values, one per configuration.
        """
        batch_size = spins.shape[0]
        return ((-1 / 2 * (torch.bmm(spins.view(batch_size, 1, self.N), (spins @ self.J).view(batch_size, self.N, 1)))[:, :, 0])
                .view(batch_size) - spins @ self.h)
