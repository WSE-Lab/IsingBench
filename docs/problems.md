# Problems

IsingBench currently implements three optimization strategies for test suite optimization.
All strategies share the same CSV input format and are accessible by name via YAML or CLI.

The Ising energy for all strategies follows the standard formulation:

$$E(\mathbf{s}) = -\sum_i h_i s_i - \frac{1}{2} \sum_{i,j} J_{ij} s_i s_j$$

where $s_i \in \{-1, +1\}$ are spin variables, with $s_i = -1$ denoting a selected test case
and $s_i = +1$ denoting an unselected test case.

---

## Input Format

All strategies expect a CSV file where each row is a test case and each column is an attribute:

```
,time,rate
0,39050.0,0.134
1,1000.0,0.096
2,47131.0,0.095
...
```

---

## WAOr — Weighted Attribute Optimization (Ratio-Based)

Adopted from [Wang et al., TSE 2024]. The fitness of each attribute is computed as the
ratio of the weighted sum of selected test cases to the total sum across all test cases.

### Fitness Function

$$f_k(\mathbf{s}) = \frac{1}{2}\left(1 + \lambda_k \frac{\sum_{i=0}^{n-1} c_i^k s_i}{\sum_{i=0}^{n-1} c_i^k}\right)$$

$$f_v = \sum_k \omega_k (f_k)^2$$

where:
- $c_i^k$ is the $k$-th attribute value of test case $i$
- $\lambda_k = +1$ for effectiveness attributes, $\lambda_k = -1$ for cost attributes
- $\omega_k$ is the user-defined weight for attribute $k$

The Ising energy $E(\mathbf{s})$ is defined as the fitness value $f_v(\mathbf{s})$.

### Parameters

| Parameter | Type | Required | Description                                                             |
|-----------|------|----------|-------------------------------------------------------------------------|
| `effectiveness` | `list[str]` | Yes | Column names treated as effectiveness attributes (higher is better)     |
| `cost` | `list[str]` | Yes | Column names treated as cost attributes (lower is better)               |
| `weights` | `dict[str, float]` | No | Per-attribute weights. Defaults to equal weight                         |
| `minimization` | `bool` | No | If `true`, adds suite size as an additional cost term. Default: `false` |

### Example

```yaml
problem:
  name: WAOr
  params:
    effectiveness:
      - rate
    cost:
      - time
    weights:
      rate: 0.3
      time: 0.7
```

```bash
--problem WAOr \
--problem-param effectiveness=['rate'] cost=['time'] weights={'rate': 0.3,'time': 0.7}
```

---

## WAOd — Weighted Attribute Optimization (Deviation-Based)

Adopted from [Wang et al., TOSEM 2024]. Shares the same input format and parameters as
WAOr, but measures the deviation of the selected subset from a theoretical optimum $L^k$
rather than the ratio.

### Fitness Function

$$f_k(\mathbf{s}) = \left(\sum_{i=0}^{n-1} c_i^k \frac{1 - s_i}{2} - L^k\right)^2$$

$$f_v = \sum_k \omega_k (f_k)^2$$

where $L^k$ is the theoretical optimum for attribute $k$:
- For **effectiveness** attributes: $L^k = \sum_i c_i^k$ (sum of all values)
- For **cost** attributes: $L^k = 0$

### Parameters

Identical to WAOr.

| Parameter | Type | Required | Description                                                             |
|-----------|------|----------|-------------------------------------------------------------------------|
| `effectiveness` | `list[str]` | Yes | Column names treated as effectiveness attributes                        |
| `cost` | `list[str]` | Yes | Column names treated as cost attributes                                 |
| `weights` | `dict[str, float]` | No | Per-attribute weights. Defaults to equal weight                         |
| `minimization` | `bool` | No | If `true`, adds suite size as an additional cost term. Default: `false` |

### Example

```yaml
problem:
  name: WAOd
  params:
    effectiveness:
      - rate
    cost:
      - time
    minimization: true
```

```bash
--problem WAOd \
--problem-param effectiveness=['rate'] cost=['time'] minimization=true
```

---

## WAOr-Budget — WAOr with Budget Constraint

Extends WAOr by incorporating a user-defined budget constraint that limits the maximum
proportion of test cases that can be selected.

The budget constraint is handled differently depending on the solver type:

- **Ising solvers (e.g. CIM)**: the constraint is embedded directly into the Ising
  Hamiltonian as a penalty term.
- **Classical solvers (e.g. GA, SA)**: the constraint is passed as a separate constraint
  function, allowing the solver to handle it independently.

### Hamiltonian (Ising solvers)

$$H = f_v(\mathbf{s}) + \alpha \left(\sum_i \frac{1 - s_i}{2} - B\right)^2$$

where $B$ is the target number of selected test cases derived from the budget percentage,
and $\alpha$ is a penalty coefficient controlling the strictness of the constraint.

### Parameters

| Parameter       | Type               | Required | Description                                                   |
|-----------------|--------------------|----------|---------------------------------------------------------------|
| `effectiveness` | `list[str]`        | Yes | Column names treated as effectiveness attributes              |
| `cost`          | `list[str]`        | Yes | Column names treated as cost attributes                       |
| `weights`       | `dict[str, float]` | No | Per-attribute weights. Defaults to `1.0` for all attributes   |
| `budget`        | `int`              | Yes | Maximum proportion of test cases to select, e.g. `10` for 10% |
| `alpha`         | `float`            | No | Penalty coefficient for the budget constraint. Default: `1.0` |

### Example

```yaml
problem:
  name: WAOr-Budget
  params:
    effectiveness:
      - rate
    cost:
      - time
    budget: 10
    alpha: 1.0
```

```bash
--problem WAOr-Budget \
--problem-param effectiveness=['rate'] cost=['time'] budget=10 alha=1.0
```

---

## References

- Wang et al. *Quantum Approximate Optimization Algorithm for Test Case Optimization.* IEEE TSE, 2024.
- Wang et al. *Test Case Minimization with Quantum Annealers.* ACM TOSEM, 2024.