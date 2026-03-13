import os.path
import time
from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np

from ..utils import IsingUtil


PROBLEM_REGISTRY = {}


def register_problem(name):
    """
    Class decorator to register a problem under a given name.

    Usage::

        @register_problem("MyProblem")
        class MyProblem(BaseProblem):
            ...

    Args:
        name: Registry key used to look up the problem class.
    """
    def decorator(cls):
        PROBLEM_REGISTRY[name] = cls
        cls.name = name
        return cls
    return decorator


class BaseProblem(ABC):
    """
        Abstract base class for all tso problems.

        Subclasses must implement the core interface methods:
        ``_calc_ising``, ``fitness_function``, ``classical_info``,
        ``get_selected_test_case``, and ``spins2solution``.

        Args:
            csv_path: Path to the CSV file containing problem instance data.
    """
    def __init__(self, csv_path):
        self.ising_calculation_time = None
        self.csv_path = csv_path

    @staticmethod
    def get_name() -> str:
        """Return the registered name of this problem."""
        pass

    def calc_ising(self, force_calc: bool = False, load_save_path: str = None, scaling: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return the Ising coupling matrix J and bias vector h.

        Results are optionally cached on disk so that expensive computations
        are only performed once.  When *scaling* is enabled the matrices are
        multiplied by the factor recommended by :func:`IsingUtil.recommend_scaling`
        and both the raw and scaled versions are persisted separately.

        Cache file layout under ``load_save_path``::

            J.npy   – unscaled coupling matrix
            h.npy   – unscaled bias vector
            Js.npy  – scaled coupling matrix  (only when scaling=True)
            hs.npy  – scaled bias vector      (only when scaling=True)

        Args:
            force_calc:     If ``True``, recompute J and h even when a cache
                            file already exists.
            load_save_path: Directory used to load or persist J and h.
                            Pass ``None`` to skip all disk I/O.
            scaling:        If ``True``, multiply J and h by the recommended
                            scaling factor before returning.

        Returns:
            J: ``(N, N)`` symmetric coupling matrix with zero diagonal.
            h: ``(N,)`` external-field bias vector.
        """
        start = None
        if load_save_path is not None:
            J_path = os.path.join(load_save_path, 'J.npy')
            h_path = os.path.join(load_save_path, 'h.npy')
            J_s_path = os.path.join(load_save_path, 'Js.npy')
            h_s_path = os.path.join(load_save_path, 'hs.npy')
            if not os.path.exists(load_save_path):
                os.makedirs(load_save_path)
            if force_calc:
                start = time.time()
                J, h = self._calc_ising()
            elif scaling:
                if os.path.exists(J_s_path) and os.path.exists(h_s_path):
                    J, h = np.load(J_s_path), np.load(h_s_path)
                else:
                    start = time.time()
                    J, h = self._calc_ising()
            elif os.path.exists(J_path) and os.path.exists(h_path):
                J, h = np.load(J_path), np.load(h_path)
            else:
                start = time.time()
                J, h = self._calc_ising()
            self.extra_load(load_save_path)
        else:
            start = time.time()
            J, h = self._calc_ising()

        if scaling:
            s = IsingUtil.recommend_scaling(J)
            J, h = s * J, s * h
            if load_save_path is not None:
                J_path = os.path.join(load_save_path, 'J.npy')
                h_path = os.path.join(load_save_path, 'h.npy')
                J_s_path = os.path.join(load_save_path, 'Js.npy')
                h_s_path = os.path.join(load_save_path, 'hs.npy')
                np.save(J_path, J / s)
                np.save(h_path, h / s)
                np.save(J_s_path, J)
                np.save(h_s_path, h)
                self.extra_save(load_save_path)
        else:
            if load_save_path is not None:
                J_path = os.path.join(load_save_path, 'J.npy')
                h_path = os.path.join(load_save_path, 'h.npy')
                np.save(J_path, J)
                np.save(h_path, h)
                self.extra_save(load_save_path)
        end = time.time()
        self.ising_calculation_time = end - start if start is not None else None
        return J, h

    @abstractmethod
    def _calc_ising(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute J and h from the problem data.

        This is the only place where the actual Ising formulation logic
        should live.  It is called by :meth:`calc_ising` whenever a valid
        cache is unavailable.

        Returns:
            J: ``(N, N)`` coupling matrix.
            h: ``(N,)`` bias vector.
        """
        pass

    @abstractmethod
    def fitness_function(self, solution):
        """
        Evaluate the objective value for a solution.

        Args:
            solution: Problem-domain solution array produced by
                      :meth:`spins2solution` or classical solvers.

        Returns:
            Scalar objective value.
        """
        pass

    @abstractmethod
    def classical_info(self):
        """
        Return the basic structural information of this problem.

        Returns:
            num_of_bits:  Number of binary decision variables.
            direction:    Optimization direction, either ``"minimize"`` or ``"maximize"``.
            constraint_fn: A callable that takes a solution array and returns
                           the number of constraint violations. Pass ``None`` if the
                           problem is unconstrained.
        """
        pass

    def fitness_function_for_spins(self, spins: np.ndarray):
        """
        Convenience wrapper: decode *spins* then evaluate the fitness.

        Args:
            spins: ``(N,)`` spin configuration with values in ``{-1, +1}``.

        Returns:
            Scalar objective value identical to
            ``self.fitness_function(self.spins2solution(spins))``.
        """
        return self.fitness_function(self.spins2solution(spins))

    @abstractmethod
    def get_selected_test_case(self, solution: np.ndarray):
        """
        Extract the subset of test cases selected by *solution*.

        Args:
            solution: Problem-domain solution array.

        Returns:
            Selected test cases in a problem-specific format.
        """
        pass

    def get_selected_test_case_for_spins(self, spins: np.ndarray):
        """
        Convenience wrapper: decode *spins* then retrieve selected test cases.

        Args:
            spins: ``(N,)`` spin configuration with values in ``{-1, +1}``.

        Returns:
            Same as ``self.get_selected_test_case(self.spins2solution(spins))``.
        """
        return self.get_selected_test_case(self.spins2solution(spins))

    @abstractmethod
    def spins2solution(self, spins: np.ndarray):
        """
        Convert an Ising spin configuration to a problem-domain solution.

        Args:
            spins: ``(N,)`` array with values in ``{-1, +1}``.

        Returns:
            Problem-domain solution array (e.g. binary selection vector).
        """
        pass

    def extra_load(self, load_save_path: str):
        """
        Hook for loading additional problem-specific artifacts from disk.

        Called by :meth:`calc_ising` immediately after the cache directory
        is accessed.  Override in subclasses that persist extra state
        (e.g. qubo, bqm).

        Args:
            load_save_path: Directory that contains the cached artifacts.
        """
        pass

    def extra_save(self, load_save_path: str):
        """
        Hook for saving additional problem-specific artifacts to disk.

        Called by :meth:`calc_ising` after J and h have been persisted.
        Override in subclasses that need to store extra state alongside
        the Ising matrices.

        Args:
            load_save_path: Directory where artifacts should be written.
        """
        pass
