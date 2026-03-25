import os
import sys
import time


def kgd_debug(*args, timestamp=False):
    frame = sys._getframe(1)
    name = frame.f_code.co_filename.split('/')[-1].split('.')[0]
    items = [f"[kgd-debug|{name}|{frame.f_code.co_name}:{frame.f_lineno}]"] + [str(a) for a in args]
    if timestamp:
        items.insert(0, f"({time.perf_counter_ns()/1_000_000_000}@{os.getpid()})")
    print(" ".join(items))
