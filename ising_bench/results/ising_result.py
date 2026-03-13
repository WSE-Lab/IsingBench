from .result import *


@dataclass
class IsingResult(Result):
    def __init__(self, best_spins_per_run: np.ndarray, lowest_energy_per_run: np.ndarray, trajectory_data: Dict[str, Any] = None):
        super().__init__(trajectory_data)
        self.best_spins_per_run = best_spins_per_run
        self.lowest_energy_per_run = lowest_energy_per_run

    def merge(self, others):
        super().merge(others)
        self.best_spins_per_run = np.concatenate((self.best_spins_per_run,
                                                  others.best_spins_per_run), 0)
        self.lowest_energy_per_run = np.concatenate((self.lowest_energy_per_run,
                                                     others.lowest_energy_per_run), 0)

    def get_lowest_energy_cross_batch(self):
        return self.lowest_energy_per_run.min()

    def get_best_spins_cross_batch(self):
        idx = self.lowest_energy_per_run.argmin()
        return self.best_spins_per_run[idx]
