import os
import sys
import time


def kgd_debug(*args, timestamp=False):
    items = [f"[kgd-debug|{parent_frame(1, sep_a='|')}]"] + [str(a) for a in args]
    if timestamp:
        items.insert(0, f"({time.perf_counter_ns()/1_000_000_000}@{os.getpid()})")
    print(" ".join(items))


def parent_frame(depth=1, sep_a=".", sep_b=":"):
    frame = sys._getframe(depth)
    name = frame.f_code.co_filename.split('/')[-1].split('.')[0]
    return f"{name}{sep_a}{frame.f_code.co_name}{sep_b}{frame.f_lineno}"
