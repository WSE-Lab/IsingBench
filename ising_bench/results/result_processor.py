import math

import numpy as np
from matplotlib import pyplot as plt

from ..results import *


class ResultProcessor:
    def __init__(self, results, previous_results, problem):
        self.results = results
        self.previous_results = previous_results
        self.problem = problem

    def calc_fitness_value_trajectory(self):
        for solver, result in self.results.items():
            if isinstance(result, IsingResult):
                if 'spins' in result.trajectory_data:
                    spins = np.transpose(result.trajectory_data['spins'], (0, 2, 1))
                    data = np.apply_along_axis(
                        self.problem.fitness_function_for_spins,
                        axis=2, arr=spins
                    )
                    result.trajectory_data['fitness_value'] = data

    def aggregation(self):
        final_results = {}
        for solver, result in self.results.items():
            single_result = {
                'execution_time': solver.execution_time,
                'seed': solver.seed
            }
            if isinstance(result, IsingResult):
                single_result['ising_model_calculation_time'] = self.problem.ising_calculation_time
                single_result['best_spins_per_run'] = result.best_spins_per_run
                single_result['best_spins'] = result.get_best_spins_cross_batch()
                single_result['lowest_energy_per_run'] = result.lowest_energy_per_run
                single_result['lowest_energy'] = result.get_lowest_energy_cross_batch()
                single_result['average_energy'] = np.average(result.lowest_energy_per_run)
                single_result['best_fitness_value_per_run'] = np.apply_along_axis(
                    self.problem.fitness_function_for_spins,
                    axis=1, arr=result.best_spins_per_run
                )
                single_result['best_fitness_value'] = self.problem.fitness_function_for_spins(
                    single_result['best_spins']
                )
                single_result['average_fitness_value'] = np.average(
                    single_result['best_fitness_value_per_run']
                )
                single_result['best_selected_test_case_per_run'] = [
                    self.problem.get_selected_test_case_for_spins(row)
                    for row in result.best_spins_per_run
                ]
                single_result['best_selected_test_case'] = self.problem.get_selected_test_case_for_spins(
                    single_result['best_spins']
                )
            elif isinstance(result, ClassicalResult):
                single_result['best_solution_per_run'] = result.best_solution_per_run
                single_result['best_solution'] = result.get_best_solution_cross_batch()
                single_result['best_fitness_value_per_run'] = result.best_fitness_value_per_run
                single_result['best_fitness_value'] = result.get_best_fitness_value_cross_batch()
                single_result['average_fitness_value'] = np.average(
                    single_result['best_fitness_value_per_run']
                )
                single_result['best_selected_test_case_per_run'] = [
                    self.problem.get_selected_test_case(row)
                    for row in result.best_solution_per_run
                ]
                single_result['best_selected_test_case'] = self.problem.get_selected_test_case(
                    single_result['best_solution']
                )
            final_results[solver.name] = single_result
        for solver, result in self.previous_results.items():
            if solver in final_results:
                continue
            final_results[solver] = result
        return final_results

    def convergence_curve(self, types):
        figs = []
        for _type in types:
            for solver, result in self.results.items():
                name = solver.name
                if _type in result.trajectory_data:
                    data = result.trajectory_data[_type]
                    if data.ndim == 2:
                        data = data[:, None, :]
                    elif data.ndim != 3:
                        print("Invalid viz data shape, must be 2D or 3D")
                        return
                    batch = data.shape[0]
                    if batch > 10:
                        data = data[:9]
                        batch = 10
                    n_col = int(math.sqrt(batch))
                    figs.append({
                        "name": f"{name}_{_type}_trajectory",
                        "fig": self.draw_convergence_curve(
                            data,
                            f"{_type} trajectory of {name}",
                            n_col,
                            False
                        )
                    })
        return figs

    def performance_comparison(self, final_results):
        data = {}
        for solver, result in final_results.items():
            data[solver] = result["best_fitness_value_per_run"]
        return self.draw_box_plot(data, "performance_comparison")

    @staticmethod
    def draw_box_plot(data: dict, title: str):
        fig, ax = plt.subplots(figsize=(8, 5))

        methods = list(data.keys())
        data = [data[m] for m in methods]

        bp = ax.boxplot(data, patch_artist=True, notch=False, vert=True)

        colors = plt.cm.Set2(np.linspace(0, 1, len(methods)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        for i, (d, color) in enumerate(zip(data, colors), start=1):
            ax.scatter([i] * len(d), d,
                       color=color, alpha=0.5, s=20, zorder=3)

        ax.set_xticks(range(1, len(methods) + 1))
        ax.set_xticklabels(methods, fontsize=11)
        ax.set_ylabel("Fitness Value")
        ax.set_title(title)
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        return fig

    @staticmethod
    def draw_convergence_curve(data: np.ndarray, title, n_cols: int = 1, show: bool = True) -> plt.Figure | None:
        batch_size, N, steps = data.shape
        if batch_size > 10:
            print("Can't visualize more than 10 figs!")
            return None
        n_rows = math.ceil(batch_size / n_cols)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3 * n_rows), sharex="all")
        axes = axes.flatten() if batch_size > 1 else [axes]

        for i in range(batch_size):
            ax = axes[i]
            for j in range(N):
                ax.plot(range(steps), data[i, j], lw=1.5)
            ax.set_title(f'Batch {i + 1}', fontsize=10)
            ax.grid(True, ls='--', alpha=0.4)
            ax.set_xlabel('Step')
            ax.set_ylabel('Value')

        for i in range(batch_size, len(axes)):
            axes[i].axis('off')

        fig.suptitle(title, fontsize=14)
        fig.tight_layout()
        if show:
            plt.show()
        return fig
