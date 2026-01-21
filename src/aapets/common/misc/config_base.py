import argparse
import dataclasses
import functools
import inspect
import logging
import sys
import typing
from abc import ABC
from argparse import Action, BooleanOptionalAction
from dataclasses import dataclass, fields
from pathlib import Path
from typing import get_origin, Annotated, get_args, Union

import yaml

logger = logging.getLogger(__name__)


T = typing.TypeVar('T', bound='IntrospectiveAbstractConfig')


class Unset:
    pass


def set_all_on(prefixes, ignore=None):
    ignore = ignore or []

    class SetAllOn(Action):
        def __init__(self, option_strings, dest, **kwargs):
            super().__init__(option_strings, dest, nargs=0, **kwargs)

        def __call__(self, parser, namespace, values, option_strings=None):
            for k in namespace.__dict__:
                if any(k.startswith(s) for s in prefixes) and not any(i in k for i in ignore):
                    setattr(namespace, k, True)
    return SetAllOn


@dataclass
class IntrospectiveAbstractConfig(ABC):
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__()
        register_yamlable_config(cls)

    def __post_init__(self):
        if len(fields(self)) == 0:
            raise TypeError(f"{self.__class__.__name__} has no fields. Did you forget the decorator?")

    @classmethod
    def fields(cls):
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

    @staticmethod
    def field_type(field):
        a_type = field.type.__args__[0]
        t_args = get_args(a_type)

        if get_origin(a_type) is Union and type(None) in t_args:
            a_type = t_args[0]

        return a_type

    @classmethod
    def populate_argparser(cls, parser):
        for field in cls.fields():
            a_type = cls.field_type(field)
            f_type, str_type = a_type, None
            default = field.default
            action = "store"

            if a_type is bool:
                f_type = None
                str_type = bool
                action = BooleanOptionalAction
            elif get_origin(a_type) is tuple:
                f_type = functools.partial(cls._parse_tuple, types=get_args(a_type))
                str_type = tuple

            if not str_type:
                str_type = f_type

            assert str_type, (
                f"Invalid user type {str_type} " f"(from {a_type=} {f_type=})"
            )

            arg_name = field.name.replace("_", "-")

            meta = list(field.type.__metadata__)
            if any(isinstance(m, dict) for m in meta):
                args = [m for m in meta if isinstance(m, dict)]
                assert len(args) == 1, f"Too many kwargs provided for {field.name} ({len(args)}, {args=})!"
                arg_kwargs = args[0]
                meta.remove(arg_kwargs)

            else:
                arg_kwargs = dict()

            assert all(isinstance(m, str) for m in meta) <= 1, "Invalid metadata, only string is allowed"

            help_kwargs = dict(default=default)
            if str_type is not bool:
                help_kwargs.update(type=str_type.__name__)
            if "choices" in arg_kwargs:
                help_kwargs.update(choices=", ".join(arg_kwargs["choices"]))
            help_msg = '. '.join([m for m in meta])
            if len(help_kwargs) > 0:
                help_msg += (
                    " ["
                    + ", ".join(f"{k}: {v}" for k, v in help_kwargs.items())
                    + "]"
                )

            kwargs = dict(
                dest=field.name,
                action=action,
                default=default,
                metavar="V",
                type=f_type,
                help=help_msg,
            )
            kwargs.update(arg_kwargs)

            parser.add_argument(
                f"--{arg_name}",
                **kwargs,
            )

    @classmethod
    def parse_command_line_arguments(cls, description):
        parser = argparse.ArgumentParser(description=description)
        cls.populate_argparser(parser)
        return parser.parse_args(namespace=cls())

    @classmethod
    def from_argparse(cls, namespace):
        data = cls()
        for field in cls.fields():
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

    def as_dict(self):
        return {field.name: getattr(self, field.name)
                for field in self.fields()}

    def where(self, **kwargs):
        """Returns a copy of this configuration object with fields adjusted as requested"""
        return dataclasses.replace(self, **kwargs)

    def override_with(self, other: "IntrospectiveAbstractConfig", verbose: bool = False):
        """Overwrites values from self with values taken from other"""
        other_fields = {f.name: getattr(other, f.name) for f in other.fields()}
        for name, value in other_fields.items():
            if hasattr(self, name):
                if verbose:
                    logger.debug(f"Overriding {name}={getattr(self, name)} with {value}")
                setattr(self, name, value)

        return self

    @classmethod
    def copy_from(cls, that: 'IntrospectiveAbstractConfig', verbose: bool = False):
        """Initializes a configuration object of this type with values taken from the argument,
         for every matching field"""
        this = cls()
        this.override_with(that, verbose)
        return this

    @classmethod
    def yaml_tag(cls): return f"!{inspect.getmodule(cls).__spec__.name}.{cls.__name__}"

    def __write(self, stream, **kwargs):
        yaml.dump(self, stream, **kwargs)

    def pretty_print(self, stream=sys.stdout, indent=2):
        self.__write(stream, indent=indent)

    def write_yaml(self, file: Path | str | typing.IO):
        if isinstance(file, Path) or isinstance(file, str):
            with open(file, "w") as f:
                self.__write(f)
        else:
            try:
                self.__write(file)
            except TypeError:
                self.__write(file, encoding="utf-8")

    @classmethod
    def read_yaml(cls: typing.Type[T], file: Path | str | typing.IO) -> T:
        def read(_f): return yaml.unsafe_load(_f)

        if isinstance(file, Path) or isinstance(file, str):
            with open(file, "r") as f:
                obj = read(f)
        else:
            obj = read(file)

        for field in obj.fields():
            if cls.field_type(field) is Path and (value := getattr(obj, field.name)) is not None:
                setattr(obj, field.name, Path(value))

        return obj


def _yaml_path(dumper, data): return dumper.represent_str(str(data))


def _config_representer(dumper: yaml.SafeDumper, config: IntrospectiveAbstractConfig) -> yaml.nodes.MappingNode:
    return dumper.represent_mapping(config.yaml_tag(), config.as_dict())


yaml.add_multi_representer(Path, _yaml_path)
yaml.add_multi_representer(IntrospectiveAbstractConfig, _config_representer)


def register_yamlable_config(cls):
    def _config_constructor(loader: yaml.SafeLoader, tag_suffix, node: yaml.nodes.MappingNode) -> cls:
        return cls(**loader.construct_mapping(node))

    yaml.add_multi_constructor(cls.yaml_tag(), _config_constructor)
    # print(f"Adding constructor yaml {cls.yaml_tag()} -> {_config_constructor}")
