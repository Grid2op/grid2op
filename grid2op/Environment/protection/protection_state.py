# Copyright (c) 2026, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

from typing import Optional

import numpy as np

from grid2op.dtypes import dt_bool, dt_int
from grid2op.Environment.protection.protection import ProtectionConfig


class ProtectionState:
    """Mutable state of the protections of a grid: one counter per protection.

    `counter[i]` is the number of consecutive steps protection `i` has been engaged
    (current on its side above its threshold). While a protection is out of service its
    counter is frozen; when it is put back in service its counter restarts from 0, so that
    a stale history never makes it trip instantly.

    .. versionadded:: 1.12.6
    """

    def __init__(self, n_prot: int):
        self.counter = np.zeros(n_prot, dtype=dt_int)
        # in service status the last time the counters were synchronized with the config
        self._last_in_service = np.ones(n_prot, dtype=dt_bool)

    @classmethod
    def from_config(cls, config: ProtectionConfig) -> "ProtectionState":
        res = cls(config.n_prot)
        res._last_in_service[:] = config.in_service
        return res

    @property
    def n_prot(self) -> int:
        return int(self.counter.shape[0])

    def reset(self, config: ProtectionConfig) -> None:
        """Set all counters to 0 (beginning of an episode)."""
        self.counter[:] = 0
        self._last_in_service[:] = config.in_service

    def sync_in_service(self, config: ProtectionConfig) -> None:
        """Reset the counters of the protections put back in service since the last call."""
        newly_in_service = config.in_service & ~self._last_in_service
        if newly_in_service.any():
            self.counter[newly_in_service] = 0
        self._last_in_service[:] = config.in_service

    def update(self, config: ProtectionConfig, engaged: np.ndarray) -> None:
        """Update the counters at the end of a step.

        Protections in service that are engaged see their counter increase by one, the other
        ones in service are reset to 0. Counters of protections out of service are frozen.
        """
        self.sync_in_service(config)
        in_service = config.in_service
        self.counter[engaged & in_service] += 1
        self.counter[~engaged & in_service] = 0

    def line_counter(self, config: ProtectionConfig, n_line: int) -> np.ndarray:
        """Per powerline maximum of the counters of the protections in service."""
        return config.line_max(self.counter, n_line, mask=config.in_service)

    def set_from_line_counter(self,
                              config: ProtectionConfig,
                              line_counter: np.ndarray,
                              engaged: Optional[np.ndarray] = None) -> None:
        """Approximate the per protection counters from a per powerline counter.

        Used when only the per powerline information is available (for example an
        observation generated with another protection configuration). Each protection
        receives the counter of its powerline, or 0 if it is not engaged (when `engaged` is given).
        """
        self.counter[:] = np.asarray(line_counter, dtype=dt_int)[config.line_id]
        if engaged is not None:
            self.counter[~engaged] = 0
        self._last_in_service[:] = config.in_service

    def copy(self) -> "ProtectionState":
        res = type(self).__new__(type(self))
        res.counter = self.counter.copy()
        res._last_in_service = self._last_in_service.copy()
        return res

    def __copy__(self) -> "ProtectionState":
        return self.copy()

    def __deepcopy__(self, memodict={}) -> "ProtectionState":
        return self.copy()
