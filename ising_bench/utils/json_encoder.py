import json

import numpy as np


class JsonEncoder(json.JSONEncoder):
    def iterencode(self, o, _one_shot=False):

        indent = self.indent or 0

        def is_1d_list(obj):
            return (
                    isinstance(obj, list)
                    and all(not isinstance(x, list) for x in obj)
            )

        def is_2d_list(obj):
            return (
                    isinstance(obj, list)
                    and obj
                    and all(isinstance(x, list) and is_1d_list(x) for x in obj)
            )

        def _iterencode(obj, level):
            pad = " " * (level * indent)
            if is_1d_list(obj):
                yield json.dumps(obj, separators=(", ", ": "))
            elif is_2d_list(obj):
                yield "["
                for i, row in enumerate(obj):
                    if i > 0:
                        yield ",\n" + pad + " " * indent
                    else:
                        yield ""
                    yield json.dumps(row, separators=(", ", ": "))
                yield "]"
            elif isinstance(obj, dict):
                yield "{\n"
                items = list(obj.items())
                for i, (k, v) in enumerate(items):
                    yield pad + " " * indent
                    yield json.dumps(k)
                    yield ": "
                    yield from _iterencode(v, level + 1)
                    if i < len(items) - 1:
                        yield ","
                    yield "\n"
                yield pad + "}"
            else:
                yield json.dumps(obj)
        return _iterencode(o, 0)


def dumps_json(obj, indent=4):
    def convert(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        elif isinstance(x, dict):
            return {k: convert(v) for k, v in x.items()}
        elif isinstance(x, list):
            return [convert(v) for v in x]
        elif isinstance(x, (np.integer, np.int_, np.int32, np.int64)):
            return int(x)
        elif isinstance(x, (np.floating, np.float32, np.float64)):
            return float(x)
        return x

    return json.dumps(
        convert(obj),
        cls=JsonEncoder,
        indent=indent,
        ensure_ascii=False
    )
