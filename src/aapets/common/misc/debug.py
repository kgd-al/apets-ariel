import sys


def kgd_debug(msg):
    frame = sys._getframe(1)
    name = frame.f_code.co_filename.split('/')[-1].split('.')[0]
    print(f"[kgd-debug|{name}|{frame.f_code.co_name}:{frame.f_lineno}] {msg}")
