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

from typing import Dict, Optional, Tuple

import numpy as np

from grid2op.dtypes import dt_bool, dt_float, dt_int
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
                    protection_threshold: float,
                    line_status: Optional[np.ndarray] = None) -> np.ndarray:
    """Whether each protection is engaged: current on its side strictly above
    ``protection_threshold * limit`` (and powerline connected when `line_status` is given).
    This does not look at `in_service`.

    `protection_threshold` is :attr:`grid2op.Parameters.Parameters.PROTECTION_THRESHOLD`."""
    flows = protection_flows(config, a_or, a_ex)
    engaged = flows > protection_threshold * config.limit
    if line_status is not None:
        engaged &= line_status[config.line_id]
    return engaged


def cascade_iteration(config: ProtectionConfig,
                      counter: np.ndarray,
                      increased: np.ndarray,
                      engaged: np.ndarray,
                      n_line: int,
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
    protections_disabled:
        Global switch (:attr:`grid2op.Parameters.Parameters.NO_OVERFLOW_DISCONNECTION`).

    Returns
    -------
    tripped_line: ``numpy.ndarray``, dtype: bool
        Powerlines to disconnect, shape (n_line,)
    """
    in_service = config.in_service
    inc = engaged & in_service & ~increased
    counter[inc] += 1
    increased |= inc
    would_trip = engaged & (counter > config.delay)
    tripped_prot = would_trip & in_service
    tripped_line = np.zeros(n_line, dtype=dt_bool)
    if protections_disabled:
        return tripped_line
    tripped_line[config.line_id[tripped_prot]] = True
    return tripped_line


# value used for "no trip scheduled" in the time before trip
NO_TRIP = -1


def compute_rho(config: ProtectionConfig,
                a_or: np.ndarray,
                a_ex: Optional[np.ndarray],
                n_line: int) -> Tuple[np.ndarray, np.ndarray]:
    """Relative loading of both sides of each powerline: the current on this side divided by the limit
    of the reference protection of this side (the protection with the lowest limit placed on it, see
    :func:`grid2op.Environment.protection.ProtectionConfig.reference_limits`).

    It is 0 on a side without protection (the reference limit is ``inf``). It does not depend on
    :attr:`grid2op.Parameters.Parameters.PROTECTION_THRESHOLD`.

    `a_ex` is only used when at least one protection is placed on the "ex" side.

    Returns
    -------
    rho_or, rho_ex: ``numpy.ndarray``, dtype: float, shape (n_line,)
    """
    ref_or, ref_ex = config.reference_limits(n_line)
    with np.errstate(divide="ignore", invalid="ignore"):
        rho_or = np.divide(a_or, ref_or)
        if config.has_ex_side:
            rho_ex = np.divide(a_ex, ref_ex)
        else:
            rho_ex = np.zeros(n_line, dtype=dt_float)
    return rho_or.astype(dt_float), rho_ex.astype(dt_float)


def steps_before_trip(config: ProtectionConfig, counter: np.ndarray) -> np.ndarray:
    """For each protection, the number of steps before it trips if its current stays above its
    threshold: ``delay + 1 - counter``. It is :data:`NO_TRIP` (-1) for the protections that are not
    engaged (counter at 0) or out of service. It is 0 for a protection that should already have
    tripped (only possible when protections are globally deactivated)."""
    engaged = (counter > 0) & config.in_service
    res = np.full(config.n_prot, NO_TRIP, dtype=dt_int)
    res[engaged] = np.maximum(config.delay[engaged] + 1 - counter[engaged], 0)
    return res


def line_reductions(config: ProtectionConfig,
                    counter: np.ndarray,
                    n_line: int) -> Dict[str, np.ndarray]:
    """Per powerline and per side summaries of the protections in service.

    Returns a dictionary with (all of shape (n_line,)):

    - ``engaged_or`` / ``engaged_ex``: maximum counter on each side (0 if none)
    - ``trip_or`` / ``trip_ex``: minimum number of steps before a trip on each side
      (:data:`NO_TRIP` if no protection is engaged on this side)
    - ``prot_trip``: :func:`steps_before_trip` of each protection, shape (n_prot,)
    """
    in_service = config.in_service
    prot_trip = steps_before_trip(config, counter)
    side = config.side_is_ex.astype(dt_int)

    engaged = np.zeros((2, n_line), dtype=dt_int)
    np.maximum.at(engaged, (side[in_service], config.line_id[in_service]), counter[in_service])

    big = np.iinfo(dt_int).max
    trip = np.full((2, n_line), big, dtype=dt_int)
    pending = prot_trip != NO_TRIP
    np.minimum.at(trip, (side[pending], config.line_id[pending]), prot_trip[pending])
    trip[trip == big] = NO_TRIP
    return {"engaged_or": engaged[0], "engaged_ex": engaged[1],
            "trip_or": trip[0], "trip_ex": trip[1],
            "prot_trip": prot_trip}


def combine_trip(trip_or: np.ndarray, trip_ex: np.ndarray) -> np.ndarray:
    """Minimum of two "steps before trip" vectors, ignoring :data:`NO_TRIP`."""
    res = np.where(trip_or == NO_TRIP, trip_ex, trip_or)
    both = (trip_or != NO_TRIP) & (trip_ex != NO_TRIP)
    res[both] = np.minimum(trip_or[both], trip_ex[both])
    return res
