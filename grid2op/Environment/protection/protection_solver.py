# Copyright (c) 2026, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

"""Vectorized trigger logic of the overcurrent protections.

Everything here works on whole arrays of protections (see
:class:`grid2op.Environment.protection.ProtectionConfig`), there is no python loop over
the protections.

.. versionadded:: 1.12.6
"""

from typing import Optional

import numpy as np

from grid2op.dtypes import dt_bool
from grid2op.Environment.protection.protection import ProtectionConfig


def protection_flows(config: ProtectionConfig,
                     a_or: np.ndarray,
                     a_ex: Optional[np.ndarray]) -> np.ndarray:
    """Current measured by each protection, shape (n_prot,).

    `a_ex` can be ``None`` when no protection is placed on the "ex" side.
    """
    if config.has_ex_side:
        return np.where(config.side_is_ex, a_ex[config.line_id], a_or[config.line_id])
    return a_or[config.line_id]


def compute_engaged(config: ProtectionConfig,
                    a_or: np.ndarray,
                    a_ex: Optional[np.ndarray],
                    thermal_limit: np.ndarray,
                    line_status: Optional[np.ndarray] = None) -> np.ndarray:
    """Whether each protection is engaged: current on its side strictly above
    ``threshold * thermal_limit`` of its powerline (and powerline connected when
    `line_status` is given). This does not look at `in_service`."""
    flows = protection_flows(config, a_or, a_ex)
    engaged = flows > config.threshold * thermal_limit[config.line_id]
    if line_status is not None:
        engaged &= line_status[config.line_id]
    return engaged


def cascade_iteration(config: ProtectionConfig,
                      counter: np.ndarray,
                      increased: np.ndarray,
                      engaged: np.ndarray,
                      n_line: int,
                      increment_counters: bool = True,
                      protections_disabled: bool = False) -> np.ndarray:
    """One iteration of the cascading failure loop.

    `counter` and `increased` are the working copies of the counters for the current
    step; they are modified in place. A counter is increased at most once per step
    (tracked with `increased`), even if the protection stays engaged over several
    iterations of the cascade.

    Order of operations: update the counters (always), compute the protections that
    would trip, then apply the gates (`in_service`, `protections_disabled`).

    Parameters
    ----------
    increment_counters:
        ``False`` for the step performed during a `reset`: counters are not increased,
        so only the instantaneous protections (``delay == 0``) can trip.

    protections_disabled:
        Global switch (:attr:`grid2op.Parameters.Parameters.NO_OVERFLOW_DISCONNECTION`).

    Returns
    -------
    tripped_line: ``numpy.ndarray``, dtype: bool
        Powerlines to disconnect, shape (n_line,)
    """
    in_service = config.in_service
    if increment_counters:
        inc = engaged & in_service & ~increased
        counter[inc] += 1
        increased |= inc
    would_trip = engaged & ((counter > config.delay) | (config.delay <= 0))
    tripped_prot = would_trip & in_service
    tripped_line = np.zeros(n_line, dtype=dt_bool)
    if protections_disabled:
        return tripped_line
    tripped_line[config.line_id[tripped_prot]] = True
    return tripped_line
