import json
import math
import numbers
import pprint
from typing import Dict, Mapping

import yaml
from colorama import Fore, Style


class EvaluationMetrics:
    def __init__(self, data: Mapping[str, float | Mapping]):
        self.data = data

    def pretty_print(self):
        print("Post-evaluation monitors:")
        pprint.pprint(self)
        print()

    def __str__(self):
        self.pretty_print()

    def __repr__(self):
        return str(self.data)

    def to_yaml(self): return yaml.dump(self.data)

    @staticmethod
    def from_yaml(yaml_data): return EvaluationMetrics(yaml.safe_load(yaml_data))

    @staticmethod
    def from_template(values: dict[str, float], template: 'EvaluationMetrics'):
        def _get(key, value):
            if isinstance(value, numbers.Number):
                return values[key]
            elif isinstance(value, Mapping):
                return _work(values)
            else:
                raise TypeError(f"Unexpected type {key}:'{type(value)} in monitors")

        def _work(subdict):
            return {key: _get(key, value) for key, value in subdict.items()}

        return EvaluationMetrics(_work(template.data))

    def keys(self):
        def _work(subdict):
            return set(subdict.keys()).union(
                _work(v) for v in subdict.values()
                if isinstance(v, Mapping)
            )
        return _work(self.data)

    @staticmethod
    def performance_compare(lhs: "EvaluationMetrics", rhs: "EvaluationMetrics", verbosity):
        width = 20
        key_width = max(len(k) for keys in
                        [["fitness"], lhs.infos or [], rhs.infos or []]
                        # [lhs.fitnesses, lhs.stats, rhs.fitnessess, rhs.stats]
                        for k in keys) + 1

        def s_format(s=''): return f"{s:{width}}"

        def f_format(f):
            if isinstance(f, numbers.Number):
                # return f"{f}"[:width-3] + "..."
                return s_format(f"{f:g}")
            else:
                return "\n" + pprint.pformat(f, width=width)

        def map_compare(lhs_d: Dict[str, float], rhs_d: Dict[str, float]):
            output, code = "", 0
            lhs_keys, rhs_keys = set(lhs_d.keys()), set(rhs_d.keys())
            all_keys = sorted(lhs_keys.union(rhs_keys))
            for k in all_keys:
                output += f"{k:>{key_width}}: "
                lhs_v, rhs_v = lhs_d.get(k), rhs_d.get(k)
                if lhs_v is None:
                    output += f"{Fore.YELLOW}{s_format()} > {f_format(rhs_v)}"
                elif rhs_v is None:
                    output += f"{Fore.YELLOW}{f_format(lhs_v)} <"
                else:
                    if lhs_v != rhs_v:
                        lhs_str, rhs_str = f_format(lhs_v), f_format(rhs_v)
                        if isinstance(lhs_v, numbers.Number):
                            diff = rhs_v - lhs_v
                            ratio = math.inf if lhs_v == 0 else diff/math.fabs(lhs_v)
                            output += f"{Fore.RED}{lhs_str} | {rhs_str}" \
                                      f"\t({diff}, {100*ratio:.2f}%)"
                        else:
                            output += "\n"
                            for lhs_item, rhs_item in zip(lhs_str.split('\n'), rhs_str.split('\n')):
                                if lhs_item != rhs_item:
                                    output += Fore.RED
                                output += f"{lhs_item:{width}s} | {rhs_item:{width}s}"
                                if lhs_item != rhs_item:
                                    output += Style.RESET_ALL
                                output += "\n"
                        code = 1
                    else:
                        output += f"{Fore.GREEN}{f_format(lhs_v)}"

                output += f"{Style.RESET_ALL}\n"
            return output, code

        def json_compliant(obj): return json.loads(json.dumps(obj))

        f_str, f_code = map_compare({"fitness": lhs.fitnesses},
                                    {"fitness": rhs.fitnesses})
        # d_str, d_code = map_compare(lhs.descriptors,
        #                             json_compliant(rhs.descriptors))
        s_str, s_code = map_compare(lhs.infos or {},
                                    json_compliant(rhs.infos or {}))

        error = max([f_code, s_code])
        verbosity += error

        max_width = max(len(line) for text in [f_str, s_str] for line in text.split('\n'))
        if verbosity == 1:
            summary = []
            codes = {0: Fore.GREEN, 1: Fore.RED}
            for _code, name in [(f_code, "fitness"),
                                # (d_code, "descriptors"),
                                (s_code, "stats")]:
                summary.append(f"{codes[_code]}{name}{Style.RESET_ALL}")
            print(f"Performance summary: {lhs.fitness} ({' '.join(summary)})")
            # print(f"Performance summary: {lhs.fitnesses} ({' '.join(summary)})")

        elif verbosity > 1:
            def header(): print("-"*max_width)
            print("Performance summary:")
            header()
            print(f_str, end='')
            header()
            # print(d_str, end='')
            # header()
            print(s_str, end='')
            header()
            print()

        # return max([f_code, d_code, s_code])
        return
