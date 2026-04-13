import numpy as np
from jmetal.core.solution import BinarySolution
from jmetal.core.problem import BinaryProblem
from jmetal.util.comparator import Comparator


class TestingProblem(BinaryProblem):
    def __init__(self, number_of_bits: int, fitness_function, direction: str, constraint_function=None):
        super().__init__()
        self.number_of_bits = number_of_bits

        if direction == "maximize":
            self.obj_directions = [self.MAXIMIZE]
        elif direction == "minimize":
            self.obj_directions = [self.MINIMIZE]
        else:
            raise ValueError("Direction must be either maximize or minimize")
        self.obj_labels = ["Ones"]
        self.fitness_function = fitness_function
        self.constraint_function = constraint_function

    def number_of_variables(self) -> int:
        return self.number_of_bits

    def number_of_objectives(self) -> int:
        return 1

    def number_of_constraints(self) -> int:
        return 1 if self.constraint_function is not None else 0

    def evaluate(self, solution: BinarySolution) -> BinarySolution:
        selection = solution.bits.astype(int).tolist()
        solution.objectives[0] = self.fitness_function(selection)
        if self.constraint_function:
            violation = self.constraint_function(solution)
            solution.constraints[0] = violation
        return solution

    def create_solution(self) -> BinarySolution:
        new_solution = BinarySolution(number_of_variables=self.number_of_variables(),
                                      number_of_objectives=self.number_of_objectives(),
                                      number_of_constraints=self.number_of_constraints(),)
        new_solution.bits = np.random.choice([True, False], size=self.number_of_bits)
        return new_solution

    def name(self) -> str:
        return "TestingProblem"


class ConstraintDominanceComparator(Comparator):
    def compare(self, solution1, solution2) -> int:
        violation1 = sum(max(c, 0) for c in solution1.constraints)
        violation2 = sum(max(c, 0) for c in solution2.constraints)

        if violation1 > violation2:
            return 1
        elif violation1 < violation2:
            return -1
        else:
            direction = solution1.problem.obj_directions[0] \
                if hasattr(solution1, 'problem') else 1
            o1 = solution1.objectives[0]
            o2 = solution2.objectives[0]
            if direction == BinaryProblem.MAXIMIZE:
                o1, o2 = -o1, -o2
            if o1 < o2:
                return -1
            elif o1 > o2:
                return 1
            return 0
