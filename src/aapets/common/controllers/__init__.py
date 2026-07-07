from .cpg import RevolveCPG
from .neighborhood_cpg import NeighborhoodCPG
from .ABCpg import ABCpg, SymmetricalABCPG

__all__ = [
    RevolveCPG,
    NeighborhoodCPG,
    ABCpg, SymmetricalABCPG
]

try:
    from .mlp_tensor import MLPTensorBrain
    __all__ += [MLPTensorBrain]
except ModuleNotFoundError:
    pass


__dict__ = {
    **{c.__name__.lower(): c for c in __all__},
    **{c.name(): c for c in __all__}
}


def get(name):
    if (controller := __dict__.get(name.lower())) is None:
        raise ValueError(f"No known controller of type {name}")
    return controller
