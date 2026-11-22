# Copyright (c) 2026, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

from __future__ import annotations

import numpy as np

from grid2op.dtypes import dt_float

from .dispatchTypes import RedispatchState


class LoadInjectionModule:
    """Computes the power that the generators have to compensate because the agent sets the consumption of some loads.

    An injection of the agent (``"injection": {"load_p": ...}``) replaces the consumption
    given by the time series. The difference is a source of imbalance, like the storage units
    or the detached elements, and has to be compensated by the generators.

    Like for the other sources, what is given to the solver is the change since the previous
    step: when the time series sets the load again (the usual case), the injection stops and
    the generators have to come back to the production of the time series.

    All the data of the step are given explicitly to :func:`LoadInjectionModule.compute`, this
    module does not read nor modify the attributes of the environment.
    """
    def __init__(self, env) -> None:
        self.env = env

    def compute(
        self,
        action,
        new_load_p: np.ndarray,
        loads_set_by_time_series: np.ndarray,
        state: RedispatchState,
        backend_action,
    ) -> dt_float:
        """
        Parameters
        ----------
        action:
            The action of the agent. Only the entries of its ``load_p`` injection that are
            finite are taken into account (``nan`` means "not modified").
        new_load_p:
            Active consumption of the loads for this step before the injections of the
            action: the one of the time series, or the one of the previous step for the loads
            that the time series does not set. It is only read.
        loads_set_by_time_series:
            Which loads have a new value in the time series this step. For these loads the
            injections of the previous steps are over.
        state:
            The dispatch state of the environment, ``load_injection_excess`` and
            ``load_injection_mw`` are updated in place
        backend_action:
            The backend action of this step, it tells which loads are detached (the injection
            on a detached load has no effect on the grid)

        Returns
        -------
        The power (in MW) that the generators have to produce in addition because of the
        change of the load injections since the previous step: positive when the loads
        consume more.
        """
        excess = state.load_injection_excess
        excess_before = dt_float(excess.sum())
        excess[loads_set_by_time_series] = 0.0
        injected_load_p = action._dict_inj.get("load_p", None)
        if injected_load_p is not None:
            modified = np.isfinite(injected_load_p)
            modified &= ~backend_action.get_load_detached()
            excess[modified] += injected_load_p[modified] - new_load_p[modified]
        excess[backend_action.get_load_detached()] = 0.0
        res = dt_float(excess.sum() - excess_before)
        state.load_injection_mw = res
        return res
