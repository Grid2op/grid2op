# Copyright (c) 2026, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import json
import os
from dataclasses import dataclass, asdict
from functools import lru_cache
from typing import Dict, Iterable, List, Literal, Optional, Sequence, Union

import numpy as np

from grid2op.dtypes import dt_bool, dt_float, dt_int
from grid2op.Exceptions import EnvError

PROTECTION_SIDES = ("or", "ex")


@dataclass(frozen=True)
class Protection:
    """Definition of one overcurrent protection, placed on one side of one powerline.

    This is only a construction / serialization unit. The environment never loops over
    :class:`Protection` objects: they are transposed into a :class:`ProtectionConfig`
    (one array per field) as soon as they are given to the environment.

    .. versionadded:: 1.12.6

    Attributes
    ----------
    line_id: ``int``
        Id of the powerline the protection is placed on.

    side: ``str``
        Side of the powerline the current is measured on: ``"or"`` (origin side, the
        `a_or` of the observation) or ``"ex"`` (extremity side, `a_ex`).

    threshold: ``float``
        Multiplier of the thermal limit of the powerline. The protection is *engaged*
        when the current on its side is strictly above ``threshold * thermal_limit``.
        As it is relative to the thermal limit, it follows the updates of the thermal
        limits (:func:`grid2op.Environment.BaseEnv.set_thermal_limit` or dynamic line rating).

    delay: ``int``
        Number of consecutive steps the protection can stay engaged without tripping.
        The powerline is disconnected when the protection is engaged and has been so for
        strictly more than ``delay`` steps. ``0`` means instantaneous.

    in_service: ``bool``
        Whether the protection is active. A protection out of service never trips and
        its counter is frozen.

    name: ``str``
        Optional name, used to refer to the protection (for example in
        :func:`grid2op.Environment.BaseEnv.set_protection_in_service`).
    """
    line_id: int
    side: Literal["or", "ex"]
    threshold: float
    delay: int
    in_service: bool = True
    name: str = ""

    def __post_init__(self):
        if self.side not in PROTECTION_SIDES:
            raise EnvError(f"The side of a protection should be one of {PROTECTION_SIDES}, found {self.side!r}")
        if int(self.delay) < 0:
            raise EnvError(f"The delay of a protection should be >= 0, found {self.delay}")
        if not float(self.threshold) > 0.:
            raise EnvError(f"The threshold of a protection should be > 0., found {self.threshold}")
        if int(self.line_id) < 0:
            raise EnvError(f"The line_id of a protection should be >= 0, found {self.line_id}")

    def to_dict(self) -> Dict:
        res = asdict(self)
        res["line_id"] = int(self.line_id)
        res["threshold"] = float(self.threshold)
        res["delay"] = int(self.delay)
        res["in_service"] = bool(self.in_service)
        return res

    @classmethod
    def from_dict(cls, dict_: Dict, name_line: Optional[Sequence[str]] = None) -> "Protection":
        """Build a protection from a dictionary (for example read from a json file).

        The powerline is given either by its id (key ``"line_id"``) or by its
        name (key ``"line_name"``, which requires `name_line`).
        """
        dict_ = dict(dict_)
        if "line_name" in dict_:
            line_name = dict_.pop("line_name")
            if name_line is None:
                raise EnvError("Impossible to use 'line_name' to define a protection without the names of the lines")
            name_line = list(name_line)
            if line_name not in name_line:
                raise EnvError(f"Unknown powerline {line_name!r} used in a protection")
            line_id = name_line.index(line_name)
            if "line_id" in dict_ and int(dict_["line_id"]) != line_id:
                raise EnvError(f"Inconsistent 'line_id' and 'line_name' for protection {dict_}")
            dict_["line_id"] = line_id
        unknown = set(dict_) - {"line_id", "side", "threshold", "delay", "in_service", "name"}
        if unknown:
            raise EnvError(f"Unknown key(s) {sorted(unknown)} to define a protection")
        return cls(line_id=int(dict_["line_id"]),
                   side=dict_.get("side", "or"),
                   threshold=float(dict_["threshold"]),
                   delay=int(dict_.get("delay", 0)),
                   in_service=bool(dict_.get("in_service", True)),
                   name=str(dict_.get("name", "")))


class ProtectionConfig:
    """All the protections of a grid, stored as a "structure of arrays".

    The structural definition (``line_id``, ``side_is_ex``, ``threshold``, ``delay``,
    ``name``) is read only. The operational status ``in_service`` is the only mutable
    array: toggling a protection does not require to rebuild the configuration.

    .. versionadded:: 1.12.6

    Attributes
    ----------
    line_id: ``numpy.ndarray``, dtype: int
        Powerline of each protection, shape (n_prot,)

    side_is_ex: ``numpy.ndarray``, dtype: bool
        ``True`` if the protection measures the current on the "ex" side, shape (n_prot,)

    threshold: ``numpy.ndarray``, dtype: float
        Threshold (multiplier of the thermal limit) of each protection, shape (n_prot,)

    delay: ``numpy.ndarray``, dtype: int
        Delay (in steps) of each protection, shape (n_prot,)

    in_service: ``numpy.ndarray``, dtype: bool
        Operational status of each protection, shape (n_prot,)

    name: ``numpy.ndarray``, dtype: str
        Name of each protection, shape (n_prot,)

    side: ``numpy.ndarray``, dtype: str
        ``"or"`` or ``"ex"`` for each protection, shape (n_prot,)
    """

    def __init__(self,
                 line_id: np.ndarray,
                 side_is_ex: np.ndarray,
                 threshold: np.ndarray,
                 delay: np.ndarray,
                 in_service: Optional[np.ndarray] = None,
                 name: Optional[np.ndarray] = None,
                 n_line: Optional[int] = None):
        line_id = np.array(line_id, dtype=dt_int).reshape(-1)
        n_prot = line_id.shape[0]
        side_is_ex = np.array(side_is_ex, dtype=dt_bool).reshape(-1)
        threshold = np.array(threshold, dtype=dt_float).reshape(-1)
        delay = np.array(delay, dtype=dt_int).reshape(-1)
        if in_service is None:
            in_service = np.ones(n_prot, dtype=dt_bool)
        in_service = np.array(in_service, dtype=dt_bool).reshape(-1)
        if name is None:
            name = np.array(["" for _ in range(n_prot)], dtype=str)
        name = np.array(name, dtype=str).reshape(-1)
        for arr_nm, arr in (("side_is_ex", side_is_ex), ("threshold", threshold), ("delay", delay),
                            ("in_service", in_service), ("name", name)):
            if arr.shape[0] != n_prot:
                raise EnvError(f"Protection config: '{arr_nm}' has {arr.shape[0]} elements "
                               f"but there are {n_prot} protections")
        if (delay < 0).any():
            raise EnvError("Protection config: all delays should be >= 0")
        if not (threshold > 0.).all():
            raise EnvError("Protection config: all thresholds should be > 0.")
        if (line_id < 0).any():
            raise EnvError("Protection config: all line ids should be >= 0")
        if n_line is not None and (line_id >= n_line).any():
            raise EnvError(f"Protection config: some protections are placed on line ids >= {n_line} "
                           f"(the number of powerlines on the grid)")
        named = name[name != ""]
        if np.unique(named).shape[0] != named.shape[0]:
            raise EnvError("Protection config: two protections have the same (non empty) name")

        self.line_id = line_id
        self.side_is_ex = side_is_ex
        self.threshold = threshold
        self.delay = delay
        self.name = name
        self.side = np.where(side_is_ex, "ex", "or")
        self._freeze()
        self.in_service = in_service
        self.has_ex_side = bool(side_is_ex.any())

    def _freeze(self) -> None:
        for arr in (self.line_id, self.side_is_ex, self.threshold, self.delay, self.name, self.side):
            arr.flags.writeable = False

    def __setstate__(self, state: Dict) -> None:
        # the read only flag is not kept by pickle
        self.__dict__.update(state)
        self._freeze()

    @property
    def n_prot(self) -> int:
        return int(self.line_id.shape[0])

    def __len__(self) -> int:
        return self.n_prot

    @classmethod
    def from_protections(cls,
                         protections: Iterable[Union[Protection, Dict]],
                         n_line: Optional[int] = None,
                         name_line: Optional[Sequence[str]] = None) -> "ProtectionConfig":
        """Transpose a list of :class:`Protection` (or of dictionaries) into a configuration."""
        prots = [el if isinstance(el, Protection) else Protection.from_dict(el, name_line=name_line)
                 for el in protections]
        return cls(line_id=[el.line_id for el in prots],
                   side_is_ex=[el.side == "ex" for el in prots],
                   threshold=[el.threshold for el in prots],
                   delay=[el.delay for el in prots],
                   in_service=[el.in_service for el in prots],
                   name=[el.name for el in prots],
                   n_line=n_line)

    @classmethod
    def from_json(cls,
                  path: Union[str, os.PathLike],
                  n_line: Optional[int] = None,
                  name_line: Optional[Sequence[str]] = None) -> "ProtectionConfig":
        """Read a configuration from a json file.

        The file contains either a list of protections, or a dictionary with a key
        ``"protections"`` holding this list. Each protection is a dictionary with the
        keys of :class:`Protection` (``"line_name"`` can replace ``"line_id"``).
        """
        with open(path, "r", encoding="utf-8") as f:
            content = json.load(f)
        return cls.from_raw(content, n_line=n_line, name_line=name_line)

    @classmethod
    def from_raw(cls,
                 content: Union["ProtectionConfig", Dict, Iterable[Union[Protection, Dict]]],
                 n_line: Optional[int] = None,
                 name_line: Optional[Sequence[str]] = None) -> "ProtectionConfig":
        """Build a configuration from any supported format (config, json content, list of protections)."""
        if isinstance(content, ProtectionConfig):
            if n_line is not None and (content.line_id >= n_line).any():
                raise EnvError(f"Protection config: some protections are placed on line ids >= {n_line} "
                               f"(the number of powerlines on the grid)")
            return content.copy()
        if isinstance(content, dict):
            if "protections" not in content:
                raise EnvError("A protection configuration given as a dictionary should have a 'protections' key")
            content = content["protections"]
        return cls.from_protections(content, n_line=n_line, name_line=name_line)

    def to_protections(self) -> List[Protection]:
        """Convert back to a list of :class:`Protection` (with the current `in_service` status)."""
        return [Protection(line_id=int(self.line_id[i]),
                           side=str(self.side[i]),
                           threshold=float(self.threshold[i]),
                           delay=int(self.delay[i]),
                           in_service=bool(self.in_service[i]),
                           name=str(self.name[i]))
                for i in range(self.n_prot)]

    def to_dict(self) -> Dict:
        return {"protections": [el.to_dict() for el in self.to_protections()]}

    def to_json(self, path: Union[str, os.PathLike]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), fp=f, indent=4)

    def copy(self) -> "ProtectionConfig":
        """Copy the configuration. Frozen arrays are shared, `in_service` is copied."""
        res = type(self).__new__(type(self))
        res.__dict__.update(self.__dict__)
        res.in_service = self.in_service.copy()
        return res

    def __copy__(self) -> "ProtectionConfig":
        return self.copy()

    def __deepcopy__(self, memodict={}) -> "ProtectionConfig":
        return self.copy()

    def __eq__(self, other) -> bool:
        if not isinstance(other, ProtectionConfig):
            return NotImplemented
        return (np.array_equal(self.line_id, other.line_id) and
                np.array_equal(self.side_is_ex, other.side_is_ex) and
                np.array_equal(self.threshold, other.threshold) and
                np.array_equal(self.delay, other.delay) and
                np.array_equal(self.in_service, other.in_service) and
                np.array_equal(self.name, other.name))

    def same_structure(self, other: "ProtectionConfig") -> bool:
        """Whether both configurations have the same protections (regardless of `in_service`)."""
        return (self.n_prot == other.n_prot and
                np.array_equal(self.line_id, other.line_id) and
                np.array_equal(self.side_is_ex, other.side_is_ex) and
                np.array_equal(self.threshold, other.threshold) and
                np.array_equal(self.delay, other.delay) and
                np.array_equal(self.name, other.name))

    def concatenate(self, other: "ProtectionConfig") -> "ProtectionConfig":
        """Return a new configuration with the protections of `self` followed by those of `other`."""
        return type(self)(line_id=np.concatenate((self.line_id, other.line_id)),
                          side_is_ex=np.concatenate((self.side_is_ex, other.side_is_ex)),
                          threshold=np.concatenate((self.threshold, other.threshold)),
                          delay=np.concatenate((self.delay, other.delay)),
                          in_service=np.concatenate((self.in_service, other.in_service)),
                          name=np.concatenate((self.name, other.name)))

    def get_ids(self, protections: Union[int, str, Sequence[Union[int, str]], np.ndarray]) -> np.ndarray:
        """Convert protection ids, names or a boolean mask into an array of protection ids."""
        if isinstance(protections, (int, np.integer, str)):
            protections = [protections]
        if isinstance(protections, np.ndarray) and protections.dtype == np.bool_:
            if protections.shape[0] != self.n_prot:
                raise EnvError(f"A mask of protections should have {self.n_prot} elements, "
                               f"found {protections.shape[0]}")
            return np.flatnonzero(protections)
        protections = list(protections)
        res = np.empty(len(protections), dtype=dt_int)
        for i, el in enumerate(protections):
            if isinstance(el, (str, np.str_)):
                ids = np.flatnonzero(self.name == el)
                if ids.shape[0] != 1:
                    raise EnvError(f"Unknown protection named {el!r}")
                res[i] = ids[0]
            else:
                el = int(el)
                if el < 0 or el >= self.n_prot:
                    raise EnvError(f"Unknown protection id {el} (there are {self.n_prot} protections)")
                res[i] = el
        return res

    def line_max(self, values: np.ndarray, n_line: int, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Maximum over the protections of each powerline of a per-protection vector (0 if no protection)."""
        res = np.zeros(n_line, dtype=values.dtype)
        if mask is None:
            np.maximum.at(res, self.line_id, values)
        else:
            np.maximum.at(res, self.line_id[mask], values[mask])
        return res


def default_from_parameters(params: "grid2op.Parameters.Parameters",
                            n_line: int,
                            in_service: Optional[np.ndarray] = None) -> ProtectionConfig:
    """Build the default configuration, the one grid2op used before protections could be configured.

    Each powerline gets two protections on its "or" side:

    - an instantaneous one (``delay=0``) at ``params.HARD_OVERFLOW_THRESHOLD`` (named ``l{i}_hard``)
    - a delayed one at ``params.SOFT_OVERFLOW_THRESHOLD`` with ``delay=params.NB_TIMESTEP_OVERFLOW_ALLOWED``
      (named ``l{i}_soft``)

    Protections ``2 * i`` and ``2 * i + 1`` are placed on powerline ``i``.

    .. versionadded:: 1.12.6
    """
    n_prot = 2 * n_line
    line_id = np.repeat(np.arange(n_line, dtype=dt_int), 2)
    threshold = np.empty(n_prot, dtype=dt_float)
    threshold[0::2] = params.HARD_OVERFLOW_THRESHOLD
    threshold[1::2] = params.SOFT_OVERFLOW_THRESHOLD
    delay = np.empty(n_prot, dtype=dt_int)
    delay[0::2] = 0
    delay[1::2] = int(params.NB_TIMESTEP_OVERFLOW_ALLOWED)
    if in_service is not None:
        in_service = np.array(in_service, dtype=dt_bool)
        if in_service.shape[0] != n_prot:
            in_service = None
    return ProtectionConfig(line_id=line_id,
                            side_is_ex=np.zeros(n_prot, dtype=dt_bool),
                            threshold=threshold,
                            delay=delay,
                            in_service=in_service,
                            name=_default_names(n_line))


@lru_cache(maxsize=8)
def _default_names(n_line: int) -> np.ndarray:
    res = np.empty(2 * n_line, dtype=object)
    res[0::2] = [f"l{i}_hard" for i in range(n_line)]
    res[1::2] = [f"l{i}_soft" for i in range(n_line)]
    res = res.astype(str)
    res.flags.writeable = False
    return res
