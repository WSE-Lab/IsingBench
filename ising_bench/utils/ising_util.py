import numpy as np
import torch

if not hasattr(np, "float_"):
    np.float_ = np.float64


class IsingUtil:
    @staticmethod
    def calc_sig(spins: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
        res = 2 * (spins > 0) - 1
        return res if isinstance(spins, np.ndarray) else res.float()

    @staticmethod
    def calc_ising_model_from_qubo_ndarray(qubo: np.ndarray):
        qubo = qubo + qubo.T - np.diag(np.diag(qubo))

        n = qubo.shape[0]
        J = np.zeros((n, n))
        h = np.zeros(n)

        for i in range(n):
            for j in range(i + 1, n):
                J[j, i] = J[i, j] = -qubo[i, j] / 4
            h[i] = -np.sum(qubo[i, :]) / 4 - qubo[i, i] / 4

        return J, h

    @staticmethod
    def recommend_scaling(J):
        eigs = np.linalg.eigvals(J)
        lam_max = np.max(np.real(eigs))
        if lam_max == 0:
            return 1
        s = 1 / lam_max
        return s


if __name__ == "__main__":
    N = 4
    qubo = np.random.random((N, N))
    qubo = np.triu(qubo)
    print(f"qubo={qubo}")
    J, h = IsingUtil.calc_ising_model_from_qubo_ndarray(qubo)
    print(f"J={J}, h={h}")
    for i in range(10):
        solution = np.random.rand(N)
        solution = (solution > 0.5) + 1 - 1
        qubo_energy = solution @ qubo @ solution
        sig = (solution * 2 - 1)
        ising_energy = -(sig @ J @ sig) / 2 - sig @ h
        print(qubo_energy - ising_energy)
