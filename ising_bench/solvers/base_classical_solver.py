from abc import ABC

from .base_solver import *


class BaseClassicalSolver(BaseSolver, ABC):
    """
    Base class for solvers that operate in the classical binary solution space.

    This class works directly with a fitness function defined over binary solution
    vectors, making it suitable for classical meta-heuristics such as genetic
    algorithms or simulated annealing.

    Args:
        fitness_function:       Callable that accepts a binary solution array of
                                shape ``(number_of_bits,)`` and returns a scalar
                                objective value.
        number_of_bits:         Length of the binary decision variable vector.
        direction:              Optimization direction, either ``"minimize"`` or
                                ``"maximize"``.
        constraint_function:    Callable that accepts a binary solution array of
                                shape ``(number_of_bits,)`` and returns the number
                                of constraint violations.
        device:                 PyTorch device string, e.g. ``"cpu"`` or ``"cuda"``.
    """
    def __init__(self, fitness_function, number_of_bits, direction,
                 constraint_function, device: str = "cpu"):
        super().__init__(device)
        self.fitness_function = fitness_function
        self.number_of_bits = number_of_bits
        self.direction = direction
        self.constraint = constraint_function
