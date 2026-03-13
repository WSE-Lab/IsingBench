from abc import ABC, abstractmethod
import random
import time

import torch

from ..results import Result


SOLVER_REGISTRY = {}


def register_solver(name):
    """
    Class decorator to register a solver under a given name.

    Usage::

        @register_solver("MySolver")
        class MySolver(BaseSolver):
            ...

    Args:
        name: Registry key used to look up the solver class.
    """
    def decorator(cls):
        SOLVER_REGISTRY[name] = cls
        cls.name = name
        return cls
    return decorator


class BaseSolver(ABC):
    """
    Abstract base class for all solvers.

    Subclasses must implement :meth:`_run`, which performs the actual
    optimization for a given batch size and returns a :class:`Result`.

    Args:
        device: PyTorch device string, e.g. ``"cpu"`` or ``"cuda"``.
    """
    def __init__(self, device="cpu"):
        self.execution_time = None
        self.seed = None
        self.device = device

    def run(self, num_runs: int = 1, batch_size: int = 1, seed: int = None)\
            -> Result:
        """
        Execute the solver for a total of *num_runs* independent runs.

        Runs are processed in batches of *batch_size* by calling :meth:`_run`
        repeatedly, then merged into a single :class:`Result`.

        A global random seed is set via ``torch.manual_seed`` before any
        computation begins. If no seed is provided, one is sampled from the
        OS entropy source so that results are non-deterministic but still
        reproducible when the same seed is passed explicitly.

        Args:
            num_runs:   Total number of independent optimization runs.
            batch_size: Number of runs processed in a single :meth:`_run` call.
            seed:       Random seed for reproducibility.  If ``None``, a seed
                        is drawn from :class:`random.SystemRandom`.

        Returns:
            A :class:`Result` containing the merged outputs of all runs.
        """
        start = time.time()
        if seed is None:
            seed = random.SystemRandom().randint(0, 2 ** 32 - 1)
        torch.manual_seed(seed)
        self.seed = seed

        with torch.no_grad():
            times = num_runs // batch_size
            last = num_runs % batch_size
            if last > 0:
                result = self._run(last)
            else:
                result = None
            for _ in range(times):
                res = self._run(batch_size)
                if result is None:
                    result = res
                else:
                    result.merge(res)
        end = time.time()
        self.execution_time = end - start
        return result

    @abstractmethod
    def _run(self, num_runs: int)\
            -> Result:
        """
        Run the solver for exactly *num_runs* parallel runs.

        This is the core method that subclasses must implement.  It is always
        called from within a ``torch.no_grad()`` context.

        Args:
            num_runs: Number of parallel runs to execute in this batch.

        Returns:
            A :class:`Result` containing the best spins and corresponding
            fitness values for each run.
        """
        pass
