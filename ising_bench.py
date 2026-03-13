from ising_bench import *

import argparse
import ast
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

import yaml
import logging
import sys
import pkgutil

for _, name, _ in pkgutil.iter_modules(solvers.__path__):
    __import__(f"ising_bench.solvers.{name}")
for _, name, _ in pkgutil.iter_modules(problems.__path__):
    __import__(f"ising_bench.problems.{name}")


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("ising_bench")
    logger.setLevel(logging.DEBUG)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S,%f"[:-3]
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

logging.getLogger("jmetal").setLevel(logging.WARNING)


@dataclass
class LibraryConfig:
    name: str
    baseline: list
    config: int


@dataclass
class BenchmarkConfig:
    custom: str
    library: LibraryConfig


@dataclass
class SolverConfig:
    name: str
    params: dict = field(default_factory=dict)
    run_params: dict = field(default_factory=dict)


@dataclass
class ResultsConfig:
    save_path: str
    convergence_curve: list[str] = field(default_factory=list)
    performance_comparison: bool = False


@dataclass
class ProblemConfig:
    name: str
    params: dict = field(default_factory=dict)
    ising_params: dict = field(default_factory=dict)


@dataclass
class ToolConfig:
    benchmark: BenchmarkConfig
    problem: ProblemConfig
    solvers: list[SolverConfig]
    results: ResultsConfig

    def to_dict(self) -> dict:
        return asdict(self)

    def to_yaml(self, path: Path = None) -> str:
        data = self.to_dict()
        result = yaml.dump(data, allow_unicode=True, sort_keys=False)
        if path:
            path.write_text(result, encoding="utf-8")
        return result


def config_from_yaml(yaml_path):
    path = Path(yaml_path)
    if not path.exists():
        raise FileNotFoundError(f"YAML file not found: {yaml_path}")

    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    b = raw["benchmark"]
    library = b.get("library", None)
    if library is not None:
        library = LibraryConfig(library["name"], library.get("baseline", []), library.get("config", -1))
    csv_path = b.get("custom", None)
    if csv_path is not None:
        csv_path = path / '..' / csv_path
    benchmark = BenchmarkConfig(csv_path, library)

    p = raw["problem"]
    problem = ProblemConfig(p["name"], p.get("params", {}), p.get("ising_params", {}))

    solvers = []
    for s in raw["solvers"]:
        solvers.append(SolverConfig(s["name"], s.get("params", {}), s.get("run_params", {})))

    r = raw["results"]
    results = ResultsConfig(path / '..' / r["save_path"], r.get("convergence_curve", []), r.get("performance_comparison", False))
    return ToolConfig(benchmark, problem, solvers, results)


def config_from_cli(args: argparse.Namespace):
    library = None
    if args.library:
        library = LibraryConfig(
            name=args.library,
            baseline=args.baseline,
            config=args.library_config,
        )
    benchmark = BenchmarkConfig(custom=args.custom, library=library)

    problem = ProblemConfig(
        name=args.problem,
        params=_parse_kvs(args.problem_param),
        ising_params=_parse_kvs(args.ising_param),
    )

    solver_params = args.solver_param or [[] for _ in args.solver]
    solver_run_params = args.solver_run_param or [[] for _ in args.solver]

    if len(solver_params) != len(args.solver):
        raise ValueError(
            f"--solver-param count ({len(solver_params)}) "
            f"must match --solver count ({len(args.solver)})"
        )

    solvers = []
    for name, params_kv, run_kv in zip(args.solver, solver_params, solver_run_params):
        solvers.append(SolverConfig(
            name=name,
            params=_parse_kvs(params_kv),
            run_params=_parse_kvs(run_kv),
        ))

    results = ResultsConfig(
        save_path=args.save_path,
        convergence_curve=args.convergence_curve,
        performance_comparison=args.performance_comparison,
    )

    return ToolConfig(benchmark=benchmark, problem=problem, solvers=solvers, results=results)


def _parse_kvs(items: list[str]) -> dict:
    result = {}
    for item in items:
        k, _, v = item.partition("=")
        result[k.strip()] = _cast(v.strip())
    return result


def _cast(value: str):
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    try:
        return json.loads(value)
    except (ValueError, json.JSONDecodeError):
        pass
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        pass
    return value


def _build_query_parser(subparsers):
    query = subparsers.add_parser(
        "query",
        help="Query available problems, solvers, benchmarks",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    query.add_argument(
        "target",
        choices=["problems", "solvers", "benchmarks", "benchmark-config"],
        help=(
            "problems        - list all supported problems\n"
            "solvers         - list all supported solvers\n"
            "benchmarks      - list all available benchmarks in library\n"
            "benchmark-config - show config of a specific benchmark\n"
        )
    )

    query.add_argument(
        "--name", metavar="NAME",
        help="Benchmark name (required for 'benchmark-config')"
    )


def _build_test_parser(subparsers):
    parser = subparsers.add_parser(
        "test",
        help="Run test suite optimization",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--yaml", metavar="FILE", help="Path to config yaml file")
    source.add_argument("--problem", metavar="NAME", help="Problem name (CLI mode)")

    # benchmark
    bench = parser.add_argument_group("Benchmark")
    bench.add_argument("--custom", metavar="PATH", help="Custom benchmark csv path")
    bench.add_argument("--library", metavar="NAME", help="Library benchmark name")
    bench.add_argument("--baseline", nargs="+", metavar="METHOD", default=[],
                       help="Baseline methods, e.g. --baseline BootQA EIDQ")
    bench.add_argument("--library-config", type=int, default=-1, metavar="INT",
                       help="Baseline config index, set this to compare with baseline")

    # problem
    prob = parser.add_argument_group("Problem")
    prob.add_argument("--problem-param", nargs="+", metavar="k=v", default=[],
                      help="Problem params, e.g. --problem-param n=10 seed=42")
    prob.add_argument("--ising-param", nargs="+", metavar="k=v", default=[],
                      help="Params for Ising model calculation, e.g. --ising-param scaling=true")

    # solvers
    solv = parser.add_argument_group("Solvers")
    solv.add_argument("--solver", metavar="NAME", action="append", default=[],
                      help="Solver name (repeatable), e.g. --solver GA --solver SA")
    solv.add_argument("--solver-param", nargs="+", action="append", metavar="k=v", default=[],
                      help="Solver params in order, e.g. --solver-param mutation=0.01 population=100")
    solv.add_argument("--solver-run-param", nargs="+", action="append", metavar="k=v", default=[],
                      help="Solver run params in order, e.g. --run-param num_runs=5")

    # results
    res = parser.add_argument_group("Results")
    res.add_argument("--save-path", metavar="DIR", default="./results",
                     help="Results save path")
    res.add_argument("--convergence-curve", nargs="+", metavar="KEY", default=[],
                     help="Keys to track, e.g. --convergence-curve spins fitness_value")
    res.add_argument("--performance-comparison", action="store_true", default=False,
                     help="Enable performance comparison plot")

    return parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="IsingBench",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    _build_query_parser(subparsers)
    _build_test_parser(subparsers)

    return parser


logger = setup_logger()


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "query":
        handle_query(args)
    if args.command == "test":
        handle_test(args)


def handle_query(args):
    if args.target == "problems":
        print("Available problems:")
        for name in PROBLEM_REGISTRY:
            print(f"  {name}")

    elif args.target == "solvers":
        print("Available solvers:")
        for name in SOLVER_REGISTRY:
            print(f"  {name}")

    elif args.target == "benchmarks":
        benchmarks = BenchmarkLibrary.list_benchmarks()
        print("Available benchmarks:")
        for b in benchmarks:
            print(f"  {b}")

    elif args.target == "benchmark-config":
        if not args.name:
            print("[ERROR] --name is required for 'benchmark-config'")
            return
        lib = BenchmarkLibrary(args.name)
        lib.summary()


def handle_test(args):
    if args.yaml is not None:
        config = config_from_yaml(args.yaml)
    else:
        config = config_from_cli(args)

    ising_solvers = []
    classical_solvers = []
    for solver in config.solvers:
        if solver.name not in SOLVER_REGISTRY:
            raise ValueError(f"cannot find solver {solver.name}")
        solver_cls = SOLVER_REGISTRY[solver.name]
        if issubclass(solver_cls, BaseClassicalSolver):
            classical_solvers.append(solver)
        elif issubclass(solver_cls, BaseIsingSolver):
            ising_solvers.append(solver)
        else:
            raise ValueError(f"solver is neither ising solver nor classical solver")

    logger.info(f"Solvers loaded │ ising={[s.name for s in ising_solvers]}"
                f" classical={[s.name for s in classical_solvers]}")

    previous_results = {}
    if config.benchmark.custom is not None:
        csv_path = config.benchmark.custom
        logger.info(f"Benchmark │ custom: {csv_path}")
    elif config.benchmark.library is not None:
        library = BenchmarkLibrary(config.benchmark.library.name)
        csv_path = library.data_file
        logger.info(f"Benchmark │ library: {config.benchmark.library.name}, file: {csv_path}")
        if config.benchmark.library.config > 0:
            new_problem_params = library.get_config(config.problem.name,
                                                    config.benchmark.library.config)
            config.problem.params = new_problem_params
            logger.debug(f"Problem params overridden by library config: {new_problem_params}")
            for baseline in config.benchmark.library.baseline:
                previous_results[baseline] = library.get_result(config.problem.name,
                                                                config.benchmark.library.config,
                                                                baseline)
            logger.info(f"Baselines loaded: {list(previous_results.keys())}")
    else:
        raise ValueError(f"Benchmark is missing")

    if config.problem.name not in PROBLEM_REGISTRY:
        raise ValueError(f"cannot find problem {config.problem.name}")

    problem = PROBLEM_REGISTRY[config.problem.name](csv_path, **config.problem.params)
    logger.info(f"Problem │ {config.problem.name} initialized")

    results = {}
    if len(ising_solvers) > 0:
        ising_params = config.problem.ising_params
        J, h = problem.calc_ising(**ising_params)
        logger.info(f"Ising model computed │ J={J.shape}, h={h.shape}")
        for solver_config in ising_solvers:
            logger.info(f"Running │ {solver_config.name} ...")
            solver = SOLVER_REGISTRY[solver_config.name](J, h, **solver_config.params)
            results[solver] = solver.run(**solver_config.run_params)
            logger.info(f"Done    │ {solver_config.name}")
    if len(classical_solvers) > 0:
        number_of_bits, direction, constraint_function = problem.classical_info()
        logger.info(f"Classical info │ bits={number_of_bits}, direction={direction}")
        for solver_config in classical_solvers:
            logger.info(f"Running │ {solver_config.name} ...")
            solver = SOLVER_REGISTRY[solver_config.name](problem.fitness_function, number_of_bits,
                                                         direction, constraint_function,
                                                         **solver_config.params)
            results[solver] = solver.run(**solver_config.run_params)
            logger.info(f"Done    │ {solver_config.name}")

    result_path = Path(config.results.save_path)
    if not result_path.exists():
        result_path.mkdir()

    result_processor = ResultProcessor(results, previous_results, problem)
    final_result = result_processor.aggregation()
    final_result_path = result_path / "result.json"

    with open(final_result_path, "w", encoding="utf-8") as f:
        f.write(dumps_json(final_result, 4))
    logger.info(f"Results saved │ {final_result_path}")

    if 'fitness_value' in config.results.convergence_curve:
        result_processor.calc_fitness_value_trajectory()

    convergence_curves = result_processor.convergence_curve(config.results.convergence_curve)
    figs_path = result_path / "figs"
    if not figs_path.exists():
        figs_path.mkdir()

    for convergence_curve in convergence_curves:
        fig_path = figs_path / f"{convergence_curve['name']}.png"
        convergence_curve["fig"].savefig(fig_path)
        logger.info(f"Fig saved │ {fig_path}")

    if config.results.performance_comparison:
        fig_path = figs_path / "performance_comparison.png"
        result_processor.performance_comparison(final_result).savefig(fig_path)
        logger.info(f"Fig saved │ {fig_path}")

    yaml_path = result_path / "config.yaml"
    config.to_yaml(yaml_path)
    logger.info(f"Config saved │ {yaml_path}")
    logger.info("All done")


if __name__ == '__main__':
    main()
