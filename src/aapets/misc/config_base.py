import ast
import functools
import logging
from abc import ABC
from dataclasses import dataclass, fields, asdict
from enum import StrEnum
from pathlib import Path
from typing import get_origin, Annotated, get_args, Union

import yaml


logger = logging.getLogger(__name__)


@dataclass
class ConfigBase(ABC):
    @classmethod
    def __fields(cls):
        return [field for field in fields(cls) if get_origin(field.type) is Annotated]

    @staticmethod
    def _parse_tuple(_str, types):
        if _str.lower() == "none":
            return None
        else:
            return tuple(
                _t(_v) for _t, _v in zip(
                    types,
                    _str.replace("(", "").replace(")", "").split(",")
                )
            )

    @classmethod
    def populate_argparser(cls, parser):
        for field in cls.__fields():
            a_type = field.type.__args__[0]
            t_args = get_args(a_type)
            f_type, str_type = a_type, None
            default = field.default
            action = "store"

            if get_origin(a_type) is Union and type(None) in t_args:
                f_type = a_type = t_args[0]

            if a_type is bool:
                f_type = ast.literal_eval
                str_type = bool
            elif get_origin(a_type) is tuple:
                f_type = functools.partial(cls._parse_tuple, types=get_args(a_type))
                str_type = tuple

            if not str_type:
                str_type = f_type

            assert str_type, (
                f"Invalid user type {str_type} " f"(from {a_type=} {f_type=})"
            )

            arg_name = field.name.replace("_", "-")
            # if (origin := cls.__origins.get(arg_name)) is not None:

            meta = list(field.type.__metadata__)
            if any(isinstance(m, dict) for m in meta):
                args = [m for m in meta if isinstance(m, dict)]
                assert len(args) == 1, f"Too many kwargs provided for {field.name} ({len(args)}, {args=})!"
                arg_kwargs = args[0]
                meta.remove(arg_kwargs)

            else:
                arg_kwargs = dict()

            assert all(isinstance(m, str) for m in meta) <= 1, "Invalid metadata, only string is allowed"

            help_kwargs = dict(default=default, type=str_type.__name__)
            if str_type is bool:
                help_kwargs.update(const="True")
                arg_kwargs.update(const="True", nargs="?")
            help_msg = (
                '. '.join([m for m in meta])
                + " ("
                + ", ".join(f"{k}: {v}" for k, v in help_kwargs.items())
                + ")"
            )
            parser.add_argument(
                f"--{arg_name}",
                action=action,
                dest=field.name,
                default=default,
                metavar="V",
                type=f_type,
                help=help_msg,
                **arg_kwargs,
            )

    @classmethod
    def from_argparse(cls, namespace):
        data = cls()
        for field in cls.__fields():
            f_name = f"{field.name}"
            attr = None
            if (
                    hasattr(namespace, f_name)
                    and (maybe_attr := getattr(namespace, f_name)) is not None
            ):
                attr = maybe_attr
            if attr is not None:
                setattr(data, field.name, attr)
        if post_init := getattr(data, "_post_init", None):  # pragma: no branch
            post_init(allow_unset=True)
        return data

    def override_with(self, other: "ConfigBase", verbose: bool = False):
        other_fields = {
            f.name: v for f in other.__fields()
            if (v := getattr(other, f.name)) != f.default
        }
        for name, value in other_fields.items():
            if hasattr(self, name):
                if verbose:
                    logger.debug(f"Overriding {name}={getattr(self, name)} with {value}")
                setattr(self, name, value)

        return self

    def write_yaml(self, path: Path):
        with open(path, "w") as f:
            yaml.dump(asdict(self), f)

    @classmethod
    def read_yaml(cls, path: Path):
        with open(path, "r") as f:
            return cls(**yaml.unsafe_load(f))
