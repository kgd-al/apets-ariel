import logging
import math
import pprint
import re
import string
from numbers import Number
from typing import Mapping, Union, Callable

import yaml
from colorama import Fore, Style


BAD = Fore.RED
WARN = Fore.YELLOW
GOOD = Fore.GREEN
RESET = Style.RESET_ALL


class EvaluationMetrics:
    Data = Mapping[str, Union[float, 'Data']]

    def __init__(self, data: Data):
        self.data = self._check_data(data)

    def pretty_print(self):
        print("Post-evaluation monitors:")
        pprint.pprint(self.data)
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
        def _work(subdict):
            return {
                key: (values[key] if isinstance(value, Number) else _work(value))
                for key, value in subdict.items()
            }

        return EvaluationMetrics(_work(template.data))

    @staticmethod
    def _check_data(data: Data) -> Mapping:
        def float_or_dict(d):
            return all(
                type(v) in {int, float} or
                (isinstance(v, Mapping) and float_or_dict(v))
                for k, v in d.items()
            )
        assert float_or_dict(data), (f"Invalid items in {pprint.pformat(data)}."
                                     f"\nOnly numbers (int/float) and sub-dictionaries are allowed")
        return data

    def keys(self):
        def _work(subdict):
            return set().union(*[
                {k} if isinstance(v, Number) else _work(v)
                for k,  v in subdict.items()
            ])
        return _work(self.data)

    def __len__(self): return len(self.data)

    @classmethod
    def compare(cls, lhs: "EvaluationMetrics", rhs: "EvaluationMetrics", verbosity):
        width = 20
        indent = "  "

        if len(lhs) == 0 and len(rhs) == 0:
            logging.warning("No metrics where provided for both ground truth and rerun")
            return

        def _key_width(_d, depth=0):
            return 0 if len(_d) == 0 else max([
                len(k)+len(indent)*depth for k in _d.keys()
            ] + [
                _key_width(_v, depth+1) for _v in _d.values() if isinstance(_v, Mapping)
            ])
        key_width = max(_key_width(lhs.data), _key_width(rhs.data)) + len(indent)

        output, errors, mismatches = cls._map_compare(
            depth=0, indent=indent,
            lhs=lhs.data, rhs=rhs.data,
            key_max_width=key_width, item_max_width=width,
        )

        verbosity += ((errors + mismatches) > 0) + 10

        if verbosity == 1:
            summary = ""
            if errors > 0:
                summary += f"{BAD}{errors} differences{RESET}"
            if mismatches > 0:
                summary += f"{WARN}{mismatches} mismatches{RESET}"
            if len(summary) == 0:
                summary += f"{GOOD}Ok"

            print(f"Performance summary:", summary)
            # print(f"Performance summary: {lhs.fitnesses} ({' '.join(summary)})")

        elif verbosity > 1:
            def non_ansi_line(_l): return re.sub(r'\x1b\[[0-9]*m', '', _l)
            max_width = max(len(non_ansi_line(line)) for line in output.split('\n')) + 1
            def header(): print("-"*max_width)

            header()
            print("Performance summary:")
            header()
            print(output)
            header()

        # return max([f_code, d_code, s_code])
        return

    @classmethod
    def _map_compare(
            cls,
            depth: int, indent: str,
            lhs: Data, rhs: Data,
            key_max_width, item_max_width,
    ):
        local_indent = depth * indent

        def flt_fmt(f: Number, align): return f"{f:{align}{item_max_width}g}"

        def fmt_obj(obj, align):
            if obj is None:
                return ""
            elif isinstance(obj, Number):
                return flt_fmt(obj, align)
            else:
                return pprint.pformat(obj, width=item_max_width)

        output, errors, missed = "", 0, 0
        lhs_keys, rhs_keys = set(lhs.keys()), set(rhs.keys())
        all_keys = sorted(lhs_keys.union(rhs_keys))
        for k in all_keys:

            lhs_v, rhs_v = lhs.get(k), rhs.get(k)
            both_numbers = (isinstance(lhs_v, Number) and isinstance(rhs_v, Number))

            align_side = "<>"[both_numbers]
            output += f"{local_indent}{k+':':{align_side}{key_max_width-len(local_indent)+1}} "

            if both_numbers:
                if lhs_v != rhs_v:
                    lhs_str, rhs_str = flt_fmt(lhs_v, '>'), flt_fmt(rhs_v, '<')
                    diff = rhs_v - lhs_v
                    ratio = math.inf if lhs_v == 0 else diff / math.fabs(lhs_v)
                    output += f"{BAD}{lhs_str:>} | {rhs_str:<} \t({diff:g}, {100 * ratio:.2f}%)"
                else:
                    lhs_str = flt_fmt(lhs_v, '<')
                    output += f"{GOOD}{lhs_str}"

            elif isinstance(lhs_v, Mapping) and isinstance(rhs_v, Mapping):
                output += "\n"
                o, e, m = cls._map_compare(
                    depth + 1, indent,
                    lhs_v, rhs_v,
                    key_max_width=key_max_width, item_max_width=item_max_width
                )
                output += o
                errors += e
                missed += m

            else:
                lhs_str, rhs_str = fmt_obj(lhs_v), fmt_obj(rhs_v)
                lhs_lines, rhs_lines = lhs_str.split("\n"), rhs_str.split("\n")
                lhs_len, rhs_len = len(lhs_lines), len(rhs_lines)
                if lhs_len < rhs_len:
                    lhs_lines.extend(["" for _ in range(rhs_len - lhs_len)])
                elif lhs_len > rhs_len:
                    rhs_lines.extend(["" for _ in range(lhs_len - rhs_len)])

                output += f"{WARN}"
                for lhs_line, rhs_line in zip(lhs_lines, rhs_lines, strict=True):
                    output += f"{'':{key_max_width}}{lhs_line} | {rhs_line}\n"

            output += f"{RESET}\n"
        return output, errors, missed
