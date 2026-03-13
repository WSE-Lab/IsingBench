# Configuration Reference

IsingBench supports two equivalent configuration modes: a **YAML file** for reproducible
experiments and **CLI flags** for quick one-off runs. Both modes produce identical behavior.

---

## YAML Configuration

### Full Schema

```yaml
benchmark:
  custom: PATH                  # path to a custom CSV file (relative to the yaml file)
  library:
    name: NAME                  # benchmark name in the library
    baseline:                   # baseline methods to load for comparison
      - MethodA
      - MethodB
    config: INT                 # config index to load (-1 = none, default: -1)

problem:
  name: NAME                    # problem encoding strategy (e.g. WAOr, WAOd, WAOr-Budget)
  params:                       # problem-specific parameters (see docs/problems.md)
    key: value
  ising_params:                 # parameters passed to calc_ising()
    force_calc: BOOL
    scaling: BOOL
    load_save_path: PATH

solvers:                        # list of solvers to run (order preserved)
  - name: NAME                  # solver name (e.g. CIM, GA, SA, BruteForce)
    params:                     # solver hyperparameters (see docs/solvers.md)
      key: value
    run_params:                 # run-time parameters
      num_runs: INT
      batch_size: INT
      seed: INT

results:
  save_path: PATH               # directory to save all outputs
  convergence_curve:            # keys to plot convergence curves for
    - STR
  performance_comparison: BOOL # whether to generate a performance comparison box plot
```

### Field Reference

#### `benchmark`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `custom` | `str` | No | Relative path to a custom CSV file. Resolved relative to the YAML file location |
| `library.name` | `str` | No | Name of a benchmark in the built-in library |
| `library.baseline` | `list[str]` | No | Baseline method names to load from the library for comparison |
| `library.config` | `int` | No | Config index to load from the library. Set `> 0` to override problem params and load baselines. Default: `-1` |

> Either `custom` or `library` must be provided.

#### `problem`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | `str` | Yes | Problem encoding strategy name. See [problems.md](problems.md) for available strategies |
| `params` | `dict` | No | Problem-specific parameters passed to the problem constructor. See [problems.md](problems.md) |
| `ising_params` | `dict` | No | Parameters passed to `calc_ising()`. See [Ising Params](#ising-params) below |

#### `solvers`

A list of solvers to run. Each entry supports:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | `str` | Yes | Solver name. See [solvers.md](solvers.md) for available solvers |
| `params` | `dict` | No | Solver hyperparameters passed to the solver constructor. See [solvers.md](solvers.md) |
| `run_params` | `dict` | No | Run-time parameters. See [Run Params](#run-params) below |

#### `results`

| Field | Type | Required | Description                                                                                                  |
|-------|------|----------|--------------------------------------------------------------------------------------------------------------|
| `save_path` | `str` | Yes      | Directory where all outputs are saved. Resolved relative to the YAML file location                           |
| `convergence_curve` | `list[str]` | No       | Keys to generate convergence curve plots for. Supported: `fitness_value`, `spins_amplitude`                  |
| `performance_comparison` | `bool` | No       | If `true`, generates a performance comparison box plot across all solvers and baselines. Defaults to `false` |

---

## CLI Configuration

### Full Usage

```bash
ising_bench test \
  # Config source (mutually exclusive, one required)
  --yaml FILE \
  --problem NAME \

  # Benchmark
  --custom PATH \
  --library NAME \
  --baseline METHOD [METHOD ...] \
  --library-config INT \

  # Problem
  --problem-param k=v [k=v ...] \
  --ising-param k=v [k=v ...] \

  # Solvers (repeatable, in order)
  --solver NAME \
  --solver-param k=v [k=v ...] \
  --solver-run-param k=v [k=v ...] \

  # Results
  --save-path DIR \
  --convergence-curve KEY [KEY ...] \
  --performance-comparison
```

### Flag Reference

#### Config Source (mutually exclusive)

| Flag | Description |
|------|-------------|
| `--yaml FILE` | Load full configuration from a YAML file |
| `--problem NAME` | Problem encoding name, enables CLI mode |

#### Benchmark

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--custom PATH` | `str` | — | Path to a custom CSV file |
| `--library NAME` | `str` | — | Benchmark name in the built-in library |
| `--baseline METHOD ...` | `list[str]` | `[]` | Baseline methods to load for comparison |
| `--library-config INT` | `int` | `-1` | Config index to load from the library |

#### Problem

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--problem-param k=v ...` | `list[str]` | `[]` | Problem-specific parameters. See [problems.md](problems.md) |
| `--ising-param k=v ...` | `list[str]` | `[]` | Ising model parameters. See [Ising Params](#ising-params) below |

#### Solvers

`--solver`, `--solver-param`, and `--solver-run-param` are **positionally matched** — the
first `--solver-param` applies to the first `--solver`, the second to the second, and so on.

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--solver NAME` | `str` | — | Solver name (repeatable). See [solvers.md](solvers.md) |
| `--solver-param k=v ...` | `list[str]` | `[]` | Hyperparameters for the corresponding solver (repeatable) |
| `--solver-run-param k=v ...` | `list[str]` | `[]` | Run-time parameters for the corresponding solver (repeatable) |

#### Results

| Flag | Type | Default | Description |
|------|------|-------|-------------|
| `--save-path DIR` | `str` | — | Directory to save all outputs |
| `--convergence-curve KEY ...` | `list[str]` | `[]`  | Keys to generate convergence curves for |
| `--performance-comparison` | `bool` | `false` | Enable performance comparison box plot |

---

## Shared Parameter Details

### Ising Params

Passed to `calc_ising()` via `ising_params` in YAML or `--ising-param` in CLI:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `scaling` | `bool` | `false` | Apply recommended scaling to J and h before solving |
| `force_calc` | `bool` | `false` | Recompute J and h even if a cached version exists |
| `load_save_path` | `str` | `None` | Directory to cache computed J and h. `None` disables caching |

### Run Params

Passed via `run_params` in YAML or `--solver-run-param` in CLI:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_runs` | `int` | `1` | Total number of independent runs |
| `batch_size` | `int` | `1` | Number of runs processed per batch |
| `seed` | `int` | `None` | Random seed. If `None`, sampled from OS entropy |

---

## Outputs

All outputs are saved under `save_path`:

```
<save_path>/
    result.json               # decoded results and fitness values for all solvers
    config.yaml               # resolved configuration used for this run
    figs/
        convergence_curve.png       # convergence curve (if requested)
        performance_comparison.png  # box plot (if requested)
```

