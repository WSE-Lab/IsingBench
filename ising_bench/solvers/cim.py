from .base_ising_solver import *
from ..results import IsingResult
from ..utils import IsingUtil


@register_solver("CIM")
class CIM(BaseIsingSolver):
    def __init__(self,
                 J: np.ndarray,
                 h: np.ndarray,
                 device: str = "cpu",
                 g2: float = 1e-3,
                 j: float = 2.0,
                 beta: float = 10.0,
                 noise_scale: float = 1.0,
                 steps: int = 1000,
                 dt: float = 2e-3):
        super().__init__(J, h, device)
        self.g2 = g2
        self.j = j
        self.beta = beta
        self.noise_scale = noise_scale
        self.steps = steps
        self.dt = dt

    @staticmethod
    def ahc_tau(p, g2):
        return p - 0.5 + np.sqrt((p - 0.5) ** 2 + p * g2 / 2.0)

    @staticmethod
    def ahc_p(t):
        return 1.0 + np.tanh((t + 2.0) / 10.0)

    def _run(self, num_runs: int) \
            -> Result:

        device = self.device
        N = self.N
        steps = self.steps

        spins_trajectory = torch.zeros(num_runs, N, steps).to(device)
        spins_amplitude_trajectory = torch.zeros(num_runs, N, steps).to(device)
        energy_trajectory = torch.zeros(num_runs, steps).to(device)
        er_trajectory = torch.zeros(num_runs, N, steps).to(device)
        nr_trajectory = torch.zeros(num_runs, N, steps).to(device)
        mr_trajectory = torch.zeros(num_runs, N, steps).to(device)

        mu = torch.zeros(num_runs, N).to(device)
        nr = torch.zeros(num_runs, N).to(device)
        mr = torch.zeros(num_runs, N).to(device)
        er = torch.ones(num_runs, N).to(device)

        sqrt_j = np.sqrt(self.j)
        mu_tilde_coe = np.sqrt(1.0 / (4.0 * self.j))

        if num_runs != 0:
            for k in range(steps):
                t = k * self.dt
                p = self.ahc_p(t)
                tau = self.ahc_tau(p, self.g2)

                spins = IsingUtil.calc_sig(mu)
                energy = self.calc_ising_energy(spins)

                spins_trajectory[:, :, k] = spins
                spins_amplitude_trajectory[:, :, k] = mu
                energy_trajectory[:, k] = energy
                er_trajectory[:, :, k] = er
                nr_trajectory[:, :, k] = nr
                mr_trajectory[:, :, k] = mr

                # Injection term
                Wr = self.noise_scale * torch.normal(0.0, 1.0, (num_runs, N), device=device)
                mu_tilde = mu + mu_tilde_coe * Wr
                inj_term = self.j * er * (torch.matmul(mu_tilde.unsqueeze(1), self.J).squeeze(1) +
                                          self.h * np.sqrt(tau / self.g2))

                # Stochastic part
                stochastic_mu = sqrt_j * (mr + nr) * Wr

                # dmudt
                dmudt = -(1 - p + self.j) * mu - self.g2 * (mu ** 2 + 2 * nr + mr) * mu + inj_term + stochastic_mu

                # derdt
                derdt = -self.beta * (self.g2 * mu_tilde ** 2 - tau) * er

                # dnrdt
                dnrdt = (-2 * (1 + self.j) * nr +
                         2 * p * mr -
                         2 * self.g2 * mu ** 2 * (2 * nr + mr) -
                         self.j * (mr + nr) ** 2)

                # dmrdt
                dmrdt = (-2 * (1 + self.j) * mr +
                         2 * p * nr -
                         2 * self.g2 * mu ** 2 * (2 * mr + nr) +
                         p - self.g2 * (mu ** 2 + mr) -
                         self.j * (mr + nr) ** 2)

                mu += dmudt * self.dt
                er += derdt * self.dt
                nr += dnrdt * self.dt
                mr += dmrdt * self.dt

        spins_trajectory = spins_trajectory.cpu().numpy()
        spins_amplitude_trajectory = spins_amplitude_trajectory.cpu().numpy()
        energy_trajectory = energy_trajectory.cpu().numpy()
        er_trajectory = er_trajectory.cpu().numpy()
        nr_trajectory = nr_trajectory.cpu().numpy()
        mr_trajectory = mr_trajectory.cpu().numpy()
        trajectory_data = {
            'spins': spins_trajectory,
            'spins_amplitude': spins_amplitude_trajectory,
            'energy': energy_trajectory,
            'er': er_trajectory,
            'nr': nr_trajectory,
            'mr': mr_trajectory,
        }

        idx = energy_trajectory.argmin(axis=1)
        best_spins_per_run = spins_trajectory[np.arange(idx.shape[0]), :, idx]
        lowest_energy_per_run = energy_trajectory[np.arange(idx.shape[0]), idx]
        result = IsingResult(best_spins_per_run, lowest_energy_per_run, trajectory_data)
        return result
