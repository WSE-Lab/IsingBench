# Solvers

IsingBench provides four built-in solvers covering both Ising-based and classical optimization
approaches. All solvers share a common interface and are accessible by name via YAML or CLI.

| Solver | Type | Description |
|--------|------|-------------|
| `CIM` | Ising-based | Simulates the physical dynamics of a Coherent Ising Machine (GAPP model) to minimize the Ising Hamiltonian |
| `BruteForce` | Ising-based | Exhaustively enumerates all $2^n$ spin configurations, guaranteed to find the global optimum |
| `GA` | Classical | Genetic Algorithm, operates in binary solution space using a fitness function |
| `SA` | Classical | Simulated Annealing, operates in binary solution space using a fitness function |

---

## CIM — Coherent Ising Machine

A software simulation of a measurement-feedback Coherent Ising Machine following the
Gaussian Approximated Positive-P (GAPP) model proposed by Inui et al. Each spin variable
is represented by an optical parametric oscillator whose amplitude evolves according to
nonlinear dynamical equations. Through iterative measurement and feedback, the coupled
oscillator network naturally converges toward low-energy configurations of the Ising Hamiltonian.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `g2` | `float` | `1e-3` | Controls the nonlinear saturation behavior of each oscillator |
| `j` | `int` | `2` | Trade-off between feedback coupling strength and coupling-induced dissipation |
| `beta` | `float` | `10.0` | Strength of amplitude homogeneity correction across spins |
| `noise_scale` | `float` | `1.0` | Magnitude of the stochastic noise term |
| `steps` | `int` | `1000` | Total number of evolution steps of the CIM dynamics |
| `dt` | `float` | `2e-3` | Integration step size of the dynamics simulation |

### Example

```yaml
solvers:
  - name: CIM
    params:
      j: 2
      beta: 10
      steps: 2000
```

```bash
--solver CIM --solver-param j=2 beta=10 steps=2000
```

---

## BruteForce

Exhaustively enumerates all $2^n$ spin configurations and returns the one with the lowest
Ising energy. Guaranteed to find the global optimum, making it useful for validating the
correctness of other solvers on small problem instances.

### Parameters

None.

### Example

```yaml
solvers:
  - name: BruteForce
```

```bash
--solver BruteForce
```

---

## GA — Genetic Algorithm

A classical evolutionary heuristic that maintains a population of binary solution vectors
and iteratively improves them through crossover and mutation operations.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `population` | `int` | `100` | Number of individuals in the population |
| `n_generations` | `int` | `100000` | Maximum number of generations |
| `crossover_rate` | `float` | `1.0` | Probability of applying crossover to a pair of individuals |
| `mutation_rate` | `float` | `0.01` | Probability of flipping each bit during mutation |

### Example

```yaml
solvers:
  - name: GA
    params:
      population: 200
      n_generations: 50000
      mutation_rate: 0.05
```

```bash
--solver GA --solver-param population=200 n_generations=50000 mutation_rate=0.05
```

---

## SA — Simulated Annealing

A classical stochastic heuristic that explores the solution space by accepting worse
solutions with a probability that decreases over time according to a cooling schedule.
The temperature decays geometrically by factor `alpha` after each evaluation until
`minimum_temperature` is reached.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `temperature` | `float` | `1.0` | Initial temperature |
| `minimum_temperature` | `float` | `1e-4` | Temperature at which the annealing stops |
| `alpha` | `float` | `0.9` | Geometric cooling rate applied after each evaluation |
| `n_evaluations` | `int` | `100000` | Maximum number of fitness evaluations |
| `mutation_rate` | `float` | `0.01` | Probability of flipping each bit when generating a neighbour |

### Example

```yaml
solvers:
  - name: SA
    params:
      temperature: 5.0
      alpha: 0.95
      n_evaluations: 50000
```

```bash
--solver SA --solver-param temperature=5.0 alpha=0.95 n_evaluations=50000
```

---

## Run Parameters

All solvers additionally accept the following run-time parameters, passed via
`--solver-run-param` in CLI or under `run_params` in YAML:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_runs` | `int` | `1` | Number of independent runs |
| `batch_size` | `int` | `1` | Number of runs processed per batch |
| `seed` | `int` | `None` | Random seed for reproducibility. If `None`, sampled from OS entropy |

### Example

```yaml
solvers:
  - name: CIM
    params:
      steps: 2000
    run_params:
      num_runs: 10
      batch_size: 5
      seed: 42
```

```bash
--solver CIM --solver-param steps=2000 --solver-run-param num_runs=10 batch_size=5 seed=42
```

---

## References

- Inui et al. *Control of Amplitude Homogeneity in Coherent Ising Machines with Artificial Zeeman Terms.* Communications Physics, 2022.
