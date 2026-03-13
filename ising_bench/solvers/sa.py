import copy

import numpy as np
from jmetal.algorithm.singleobjective import SimulatedAnnealing
from jmetal.operator.mutation import BitFlipMutation
from jmetal.util.termination_criterion import StoppingByEvaluations
from jmetal.core.solution import BinarySolution
from jmetal.core.problem import BinaryProblem

from .base_classical_solver import *
from .testing_problem import TestingProblem
from ..results import ClassicalResult


@register_solver("SA")
class SA(BaseClassicalSolver):
    def __init__(self, fitness_function, number_of_bits, direction, constraint_function,
                 temperature: float = 1.0, minimum_temperature: float = 0.0001,
                 alpha: float = 0.9, n_evaluations: int = 100000,
                 mutation_rate: float = 0.01):
        super().__init__(fitness_function, number_of_bits, direction, constraint_function)
        self.problem = TestingProblem(number_of_bits, fitness_function, direction, constraint_function)
        self.temperature = temperature
        self.minimum_temperature = minimum_temperature
        self.alpha = alpha
        self.n_evaluations = n_evaluations
        self.mutation_rate = mutation_rate

    def _run(self, num_runs: int) -> ClassicalResult:
        if num_runs > 1:
            raise NotImplementedError('Multiple runs for SA not implemented yet.')
        algorithm = ConstraintAwareSA(
            problem=self.problem,
            mutation=BitFlipMutation(self.mutation_rate),
            termination_criterion=StoppingByEvaluations(max_evaluations=self.n_evaluations),
        )

        algorithm.temperature = self.temperature
        algorithm.minimum_temperature = self.minimum_temperature
        algorithm.alpha = self.alpha
        algorithm.run()
        best_fitness_value = algorithm.result().objectives[0]
        best_solution = np.array([1 if bit else 0 for bit in algorithm.result().variables])
        return ClassicalResult(np.array([best_solution]), np.array([best_fitness_value]))


class ConstraintAwareSA(SimulatedAnnealing):

    def _violation(self, solution) -> float:
        return sum(max(c, 0) for c in solution.constraints)

    def step(self) -> None:
        mutated_solution = copy.deepcopy(self.solutions[0])
        mutated_solution = self.mutation.execute(mutated_solution)
        mutated_solution = self.evaluate([mutated_solution])[0]

        current = self.solutions[0]
        current_violation = self._violation(current)
        new_violation = self._violation(mutated_solution)

        if new_violation < current_violation:
            self.solutions[0] = mutated_solution
        elif new_violation > current_violation:
            pass
        else:
            acceptance = self.compute_acceptance_probability(
                current.objectives[0],
                mutated_solution.objectives[0],
                self.temperature,
            )
            if acceptance > random.random():
                self.solutions[0] = mutated_solution

        self.temperature *= self.alpha
