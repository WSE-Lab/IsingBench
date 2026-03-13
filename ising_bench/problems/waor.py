from .base_problem import *

from typing import Tuple
import numpy as np
import pandas as pd


@register_problem("WAOr")
class WAOr(BaseProblem):
    def __init__(self, csv_path: str, effectiveness: list[str], cost: list[str],
                 weights: dict[str, int] = None, minimization: bool = True):
        super().__init__(csv_path)
        self.csv_path = csv_path
        self.minimization = minimization
        self.effective_list = effectiveness.copy()
        self.cost_list = cost.copy()
        # add minimization
        if minimization:
            self.cost_list.append('minimization')

        # check intersection
        inter = list(set(self.effective_list).intersection(self.cost_list))
        if len(inter) > 0:
            raise ValueError("effective and cost lists should not contain duplicate item")
        # calc equal weight
        union = list(set(self.effective_list).union(self.cost_list))
        if weights is None:
            num = len(union)
            weight = 1.0 / num
            weights = {e: weight for e in union}
        self.weights = weights
        # check weight setting
        if set(union) != set(self.weights.keys()):
            raise ValueError("weights setting error")
        if np.array(list(weights.values())).sum() != 1:
            raise ValueError("the sum of weights should be equal to 1")

        df = pd.read_csv(self.csv_path)
        self.N = df.shape[0]
        self.index = np.array(df[df.columns[0]].tolist())
        header = df.columns.tolist()
        # add minimization to df
        header.append("minimization")
        df['minimization'] = 1

        for item in self.effective_list + self.cost_list:
            if item not in header:
                raise ValueError(f"{item} is not in header")
        self.data = df[list(set(self.effective_list + self.cost_list))].to_dict("list")

    def _calc_ising(self) -> Tuple[np.ndarray, np.ndarray]:
        J = np.zeros((self.N, self.N), dtype=float)
        h = np.zeros(self.N, dtype=float)
        union = list(set(self.effective_list).union(self.cost_list))
        for item in union:
            sign = 1 if item in self.effective_list else -1
            data = np.array(self.data[item])
            J -= self.weights[item] * np.outer(data, data) / data.sum() ** 2
            h -= sign * self.weights[item] * data / data.sum()
        np.fill_diagonal(J, 0)
        return J, h

    def classical_info(self):
        return self.N, "minimize", None

    def fitness_function(self, solution: np.ndarray):
        sig = np.ones_like(solution) - np.array(solution) * 2
        union = list(set(self.effective_list).union(self.cost_list))
        value = 0
        for item in union:
            sign = 1 if item in self.effective_list else -1
            data = np.array(self.data[item])
            value += self.weights[item] * ((1 + sign * data.dot(sig) / data.sum()) / 2) ** 2
        return value

    def get_selected_test_case(self, solutions: np.ndarray):
        return self.index[np.where(solutions == 1)]

    def spins2solution(self, spins: np.ndarray):
        return (1 - spins) / 2
