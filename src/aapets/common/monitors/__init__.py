import importlib
import inspect
import logging
import pkgutil
import sys
from pathlib import Path

from ._monitor import MonitorBase

# All other imports are done automatically


def monitor_by_suffixes(suffixes: list[str], package=None) -> dict[str, type]:
    """
    Import any sub-package and returns class therein defined,
     if they match any of the provided suffixes
    :param suffixes: list of suffixes for class
    :param package: where to look (defaults to here)
    :return: dictionary of classes with matching names
    """
    result = {}
    if package is None:
        package = sys.modules[__name__]
    package_path = Path(package.__file__).parent

    for py_file in package_path.rglob("*.py"):
        if py_file.name == "__init__.py":
            continue
        # Convert path to dotted module name
        relative = py_file.relative_to(package_path.parent)
        module_name = "." + ".".join(relative.with_suffix("").parts[1:])

        try:
            module = importlib.import_module(module_name, package=package.__name__)
        except ImportError:
            logging.error("Error importing", module_name)
            continue

        for name, cls in inspect.getmembers(module, inspect.isclass):
            if cls.__module__ == module.__name__ and any(cls.__name__.endswith(s) for s in suffixes):
                result[cls.__name__] = cls

    return result


_registry = monitor_by_suffixes(["Monitor", "Plotter"])
globals().update(_registry)
__all__ = list(_registry.keys())
_registry = {c.name(): c for c in _registry.values()}


def metrics(_name, /, *args, **kwargs):
    if (monitor := _registry.get(_name.lower())) is None:
        raise ValueError(f"No known monitor of type {_name}")
    return monitor(*args, **kwargs)
