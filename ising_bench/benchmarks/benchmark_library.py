import json
import pandas as pd
from pathlib import Path

import yaml


class BenchmarkLibrary:
    def __init__(self, benchmark_name: str):
        self.benchmark_name = benchmark_name
        self.base_path = Path(__file__).parent
        self.benchmark_path = self.base_path / benchmark_name

        if not self.benchmark_path.exists():
            raise FileNotFoundError(f"Benchmark folder not found: {self.benchmark_path}")

        self.info = self._load_info()
        self.data_file = self.benchmark_path / self.info["file"]

    def _load_info(self) -> dict:
        info_path = self.benchmark_path / "info.json"
        if not info_path.exists():
            raise FileNotFoundError(f"info.json not found in {self.benchmark_path}")
        with open(info_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def list_benchmarks() -> list[str]:
        base_path = Path(__file__).parent
        return [p.name for p in base_path.iterdir() if (p / "info.json").exists()]
    def list_problems(self) -> list[str]:
        return list(self.info["results"].keys())

    def list_configs(self, problem: str) -> list[str]:
        if problem not in self.info["results"]:
            raise KeyError(f"Problem '{problem}' not found. Available: {self.list_problems()}")
        return list(self.info["results"][problem].keys())

    def list_methods(self, problem: str, config: int) -> list[str]:
        if str(config) not in self.list_configs(problem):
            raise KeyError(f"Config '{config}' not found under '{problem}'. Available: {self.list_configs(problem)}")
        return self.info["results"][problem][str(config)]

    def get_result(self, problem: str, config: int, method: str):
        available = self.list_methods(problem, config)
        if method not in available:
            raise KeyError(f"Method '{method}' not found under '{problem}' config {config}. Available: {available}")

        result_path = self.benchmark_path / problem / str(config) / f"{method}.json"
        if not result_path.exists():
            raise FileNotFoundError(f"Result file not found: {result_path}")

        return json.load(open(result_path, "r", encoding="utf-8"))

    def get_config(self, problem: str, config: int) -> pd.DataFrame:
        available = self.list_configs(problem)
        if str(config) not in available:
            raise KeyError(f"Config '{config}' not found under '{problem}'. Available: {available}")

        config_path = self.benchmark_path / problem / str(config) / "config.yaml"
        return yaml.safe_load(open(config_path, "r", encoding="utf-8"))

    def summary(self):
        print(f"Benchmark : {self.benchmark_name}")
        print(f"Data file : {self.info['file']}")
        print()

        for problem, configs in self.info["results"].items():
            print(f"  [{problem}]")
            for config_id, methods in configs.items():
                print(f"    config {config_id}")
                for m in methods:
                    print(f"      - {m}")
            print()

    def get_config_yaml(self, metric: str, config_id: str) -> dict:
        """读取某 metric + config 对应的 yaml"""
        config_path = self.benchmark_path / "configs" / f"{metric}_{config_id}.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Config yaml not found: {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
