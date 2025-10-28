import numpy as np
from mujoco import MjModel, MjData


def bounded_function(l, u):
    def wrapper(f):
        f.bounds = (l, u)
        return f
    return wrapper


@bounded_function(0, 5)
def get_speed(model: MjModel, data: MjData, tracked: dict) -> float:
    return (
        float(np.sqrt(sum(v ** 2 for v in tracked["apet1_core"].xpos))) / data.time
        if data.time > 0 else 0
    )


@bounded_function(0, 4)
def get_weight(model: MjModel, data: MjData, tracked: dict) -> float:
    return float(model.body("apet1_core").subtreemass[0])
