from .cpg import RevolveCPG

__all__ = [
    RevolveCPG,
]


__dict__ = {
    **{c.__name__.lower(): c for c in __all__},
    **{c.name(): c for c in __all__}
}


def get(name):
    if (controller := __dict__.get(name.lower())) is None:
        raise ValueError(f"No known controller of type {name}")
    return controller
