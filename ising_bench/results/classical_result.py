from .result import *


@dataclass
class ClassicalResult(Result):
    def __init__(self, best_solution_per_run: np.ndarray, best_fitness_value_per_run: np.ndarray,
                 trajectory_data: Dict[str, Any] = None):
        super().__init__(trajectory_data)
        self.best_solution_per_run = best_solution_per_run
        self.best_fitness_value_per_run = best_fitness_value_per_run

    def merge(self, others):
        super().merge(others)
        self.best_solution_per_run = np.concatenate((self.best_solution_per_run,
                                                     others.best_solution_per_run), 0)
        self.best_fitness_value_per_run = np.concatenate((self.best_fitness_value_per_run,
                                                          others.best_fitness_value_per_run), 0)

    def get_best_solution_cross_batch(self):
        idx = self.best_fitness_value_per_run.argmin()
        return self.best_solution_per_run[idx]

    def get_best_fitness_value_cross_batch(self):
        return self.best_fitness_value_per_run.min()
