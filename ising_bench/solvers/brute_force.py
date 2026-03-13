from .base_ising_solver import *
from ..results import IsingResult

from itertools import product
import torch


@register_solver("BruteForce")
class BruteForce(BaseIsingSolver):

    def _run(self, num_runs: int) -> Result:
        best_energy = torch.full((num_runs,), torch.inf)
        best_spins = torch.full((num_runs, self.N), torch.inf)

        for bits in product([-1, 1], repeat=self.N):
            spins = torch.asarray(bits, dtype=torch.float).to(self.device)
            energy = self.calc_ising_energy(spins.repeat(num_runs, 1))

            improved = energy < best_energy
            best_energy = torch.where(improved, energy, best_energy)
            best_spins[improved] = spins

        return IsingResult(best_spins.cpu().numpy(), best_energy.cpu().numpy())
