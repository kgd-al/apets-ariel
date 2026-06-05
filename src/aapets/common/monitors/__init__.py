from ._monitor import Monitor

from .behavioral import XSpeedMonitor, YSpeedMonitor, XYSpeedMonitor, KernelRewardMonitor, GymRewardMonitor, \
    GymAntKernelRewardMonitor, GymAntGymRewardMonitor
from .morphological import WeightMonitor

from .plotters.trajectory import TrajectoryPlotter
from .plotters.brain_activity import BrainActivityPlotter

metrics = [
    WeightMonitor,
    XSpeedMonitor, YSpeedMonitor, XYSpeedMonitor,
    KernelRewardMonitor, GymRewardMonitor,
    GymAntKernelRewardMonitor, GymAntGymRewardMonitor,
]


__all__ = [
    TrajectoryPlotter,
    BrainActivityPlotter,
    *metrics
]


__dict__ = {c.name(): c for c in __all__}


def metrics(_name, /, *args, **kwargs):
    if (monitor := __dict__.get(_name.lower())) is None:
        raise ValueError(f"No known monitor of type {_name}")
    return monitor(*args, **kwargs)
