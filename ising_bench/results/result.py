from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Any
import numpy as np


@dataclass
class Result:
    def __init__(self, trajectory_data: Dict[str, Any]):
        if trajectory_data is None:
            self.trajectory_data = defaultdict()
        else:
            self.trajectory_data = trajectory_data

    def merge(self, others):
        for item in self.trajectory_data.items():
            key = item[0]
            if key not in others.trajectory_data:
                continue
            value = np.concatenate((item[1], others.trajectory_data[key]), 0)
            self.trajectory_data[key] = value
        for item in others.trajectory_data.items():
            key = item[0]
            if key not in self.trajectory_data:
                self.trajectory_data[key] = item[1]
