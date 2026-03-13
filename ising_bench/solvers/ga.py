import numpy as np
from jmetal.algorithm.singleobjective import GeneticAlgorithm
from jmetal.operator.crossover import SPXCrossover
from jmetal.operator.mutation import BitFlipMutation
from jmetal.util.termination_criterion import StoppingByEvaluations

from .base_classical_solver import *
from .testing_problem import TestingProblem, ConstraintDominanceComparator
from ..results import ClassicalResult


@register_solver("GA")
class GA(BaseClassicalSolver):
    def __init__(self, fitness_function, number_of_bits, direction, constraint_function,
                 population: int = 100, n_generations: int = 100000,
                 crossover_rate: float = 1.0, mutation_rate: float = 0.01):
        super().__init__(fitness_function, number_of_bits, direction, constraint_function)
        self.problem = TestingProblem(number_of_bits, fitness_function, direction, constraint_function)
        self.population = population
        self.n_generations = n_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate

    def _run(self, num_runs: int) -> Result:
        if num_runs > 1:
            raise NotImplementedError('Multiple runs for GA not implemented yet.')
        algorithm = GeneticAlgorithm(
            problem=self.problem,
            population_size=self.population,
            offspring_population_size=self.population,
            mutation=BitFlipMutation(self.mutation_rate),
            crossover=SPXCrossover(self.crossover_rate),
            termination_criterion=StoppingByEvaluations(max_evaluations=self.n_generations),
            solution_comparator=ConstraintDominanceComparator(),
        )
        algorithm.run()
        best_fitness_value = algorithm.result().objectives[0]
        best_solution = np.array([1 if bit else 0 for bit in algorithm.result().variables])
        return ClassicalResult(np.array([best_solution]), np.array([best_fitness_value]))
