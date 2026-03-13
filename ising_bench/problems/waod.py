from .base_problem import *

import os
from typing import Tuple
import dimod
import numpy as np
import pandas as pd


@register_problem("WAOd")
class WAOd(BaseProblem):
    def __init__(self, csv_path: str, effectiveness: list[str], cost: list[str],
                 weights: dict[str, int] = None, minimization: bool = False):
        super().__init__(csv_path)
        self.minimization = minimization
        # add minimization
        self.effective_list = effectiveness.copy()
        self.cost_list = cost.copy()
        if minimization:
            self.cost_list.append('minimization')

        # check intersection
        inter = list(set(effectiveness).intersection(self.cost_list))
        if len(inter) > 0:
            raise ValueError("effective and cost lists should not contain duplicate item")
        # calc equal weight
        union = list(set(effectiveness).union(self.cost_list))
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
        df = df.drop(df[df['rate'] == 0].index)
        self.N = df.shape[0]
        self.actual_index = np.array(df[df.columns[0]].tolist())
        header = df.columns.tolist()
        # add minimization to df
        header.append("minimization")
        df['minimization'] = 1

        for item in effectiveness + self.cost_list:
            if item not in header:
                raise ValueError(f"{item} is not in header")
        self.data = df[list(set(effectiveness + self.cost_list))].to_dict("list")
        self.bqm = None

    def create_bqm(self):
        if self.bqm is not None:
            return self.bqm
        bqms = 0
        union = list(set(self.effective_list).union(self.cost_list))
        for item in union:
            data = np.array(self.data[item])
            data_dic = {i: d for i, d in enumerate(data)}
            limit = data.sum() if item in self.effective_list else 0
            bqm = dimod.BinaryQuadraticModel(data_dic, {}, 0, dimod.Vartype.BINARY)
            if item == "time":
                bqm.normalize()
            bqms += self.weights[item] * pow((bqm - limit) / self.N, 2)

        self.bqm = bqms
        return bqms

    def _calc_ising(self) -> Tuple[np.ndarray, np.ndarray]:
        bqms = self.create_bqm()

        h = np.zeros(self.N)
        J = np.zeros((self.N, self.N))
        linear, quadratic, self.offset = bqms.to_ising()
        for v, bias in linear.items():
            h[v] = bias

        for (i, j), coupling in quadratic.items():
            J[i][j] = coupling
            J[j][i] = coupling
        return -J, h

    def classical_info(self):
        return self.N, "minimize", None

    def fitness_function(self, solution: np.ndarray):
        bqms = self.create_bqm()
        return bqms.energy(solution)

    def get_selected_test_case(self, solutions: np.ndarray):
        return self.actual_index[np.where(solutions == 1)]

    def spins2solution(self, spins: np.ndarray):
        return (1 - spins) / 2

    def extra_load(self, load_save_path: str):
        self.bqm = dimod.BQM.from_file(open(os.path.join(load_save_path, "model.bqm"), "rb"))

    def extra_save(self, load_save_path: str):
        file = self.bqm.to_file()
        path = os.path.join(load_save_path, "model.bqm")
        with open(path, "wb") as f:
            f.write(file.read())
