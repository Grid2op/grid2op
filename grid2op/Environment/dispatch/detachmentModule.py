# Copyright (c) 2026, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

from __future__ import annotations

from grid2op.dtypes import dt_float

from .dispatchTypes import DetachmentResult, RedispatchState


class DetachmentModule:
    def __init__(self, env) -> None:
        self.env = env

    def feed_data(self, new_p_th, state: RedispatchState) -> None:
        self.env._prev_gen_p[:] = new_p_th
        self.env._aux_retrieve_modif_act(self.env._prev_load_p, self.env._env_modification, "load_p")
        self.env._aux_retrieve_modif_act(self.env._prev_load_q, self.env._env_modification, "load_q")

    def compute(
        self,
        new_p,
        new_p_th,
        state: RedispatchState,
        backend_action=None,
    ) -> DetachmentResult:
        if backend_action is None:
            backend_action = self.env._backend_action
        gen_detached_user = backend_action.get_gen_detached()
        load_detached_user = backend_action.get_load_detached()

        mw_gen_lost_this = new_p[gen_detached_user].sum()
        mw_load_lost_this = self.env._prev_load_p[load_detached_user].sum()
        total_power_lost = -mw_gen_lost_this + mw_load_lost_this
        state.detached_elements_mw = dt_float(
            -total_power_lost
            + state.actual_dispatch[gen_detached_user].sum()
            - state.detached_elements_mw_prev
        )
        state.detached_elements_mw_prev = dt_float(-total_power_lost)

        new_p[gen_detached_user] = 0.0
        new_p_th[gen_detached_user] = 0.0
        state.actual_dispatch[gen_detached_user] = 0.0
        return DetachmentResult(
            detached_mw=state.detached_elements_mw,
            gen_detached=gen_detached_user,
            new_p=new_p,
            new_p_th=new_p_th,
        )
