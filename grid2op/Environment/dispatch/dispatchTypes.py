# Copyright (c) 2026, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

import numpy as np

from grid2op.dtypes import dt_bool, dt_float, dt_int

if TYPE_CHECKING:
    from grid2op.Environment.baseEnv import BaseEnv


@dataclass
class StorageResult:
    amount_storage_mw: np.float64 = dt_float(0.0)
    storage_power: Optional[np.ndarray] = None


@dataclass
class CurtailmentResult:
    sum_curtailment_mw: np.float64 = dt_float(0.0)
    gen_curtailed: Optional[np.ndarray] = None


@dataclass
class DetachmentResult:
    detached_mw: np.float64 = dt_float(0.0)
    gen_detached: Optional[np.ndarray] = None
    new_p: Optional[np.ndarray] = None
    new_p_th: Optional[np.ndarray] = None


@dataclass
class GuardInfo:
    total_curtailment: np.float64 = dt_float(0.0)
    total_storage: np.float64 = dt_float(0.0)
    updated_env_action: bool = False


@dataclass
class RedispatchState:
    target_dispatch: Optional[np.ndarray] = None
    already_modified_gen: Optional[np.ndarray] = None
    actual_dispatch: Optional[np.ndarray] = None
    gen_uptime: Optional[np.ndarray] = None
    gen_downtime: Optional[np.ndarray] = None
    gen_activeprod_t: Optional[np.ndarray] = None
    gen_activeprod_t_redisp: Optional[np.ndarray] = None
    storage_current_charge: Optional[np.ndarray] = None
    storage_previous_charge: Optional[np.ndarray] = None
    action_storage: Optional[np.ndarray] = None
    amount_storage: np.float64 = dt_float(0.0)
    amount_storage_prev: np.float64 = dt_float(0.0)
    storage_power: Optional[np.ndarray] = None
    storage_power_prev: Optional[np.ndarray] = None
    limit_curtailment: Optional[np.ndarray] = None
    limit_curtailment_prev: Optional[np.ndarray] = None
    gen_before_curtailment: Optional[np.ndarray] = None
    sum_curtailment_mw: np.float64 = dt_float(0.0)
    sum_curtailment_mw_prev: np.float64 = dt_float(0.0)
    detached_elements_mw: np.float64 = dt_float(0.0)
    detached_elements_mw_prev: np.float64 = dt_float(0.0)
    limited_before: np.float64 = dt_float(0.0)

    @classmethod
    def allocate(cls, n_gen: int, n_storage: int) -> "RedispatchState":
        return cls(
            target_dispatch=np.zeros(n_gen, dtype=dt_float),
            already_modified_gen=np.zeros(n_gen, dtype=dt_bool),
            actual_dispatch=np.zeros(n_gen, dtype=dt_float),
            gen_uptime=np.zeros(n_gen, dtype=dt_int),
            gen_downtime=np.zeros(n_gen, dtype=dt_int),
            gen_activeprod_t=np.zeros(n_gen, dtype=dt_float),
            gen_activeprod_t_redisp=np.zeros(n_gen, dtype=dt_float),
            storage_current_charge=np.zeros(n_storage, dtype=dt_float),
            storage_previous_charge=np.zeros(n_storage, dtype=dt_float),
            action_storage=np.zeros(n_storage, dtype=dt_float),
            storage_power=np.zeros(n_storage, dtype=dt_float),
            storage_power_prev=np.zeros(n_storage, dtype=dt_float),
            limit_curtailment=np.ones(n_gen, dtype=dt_float),
            limit_curtailment_prev=np.ones(n_gen, dtype=dt_float),
            gen_before_curtailment=np.zeros(n_gen, dtype=dt_float),
        )


@dataclass(frozen=True)
class RedispatchConstraints:
    new_p: np.ndarray
    gen_activeprod_t: np.ndarray
    target_dispatch: np.ndarray
    gen_participating: np.ndarray
    amount_storage_mw: np.float64
    sum_curtailment_mw: np.float64
    detached_mw: np.float64
    pmin: np.ndarray
    pmax: np.ndarray
    ramp_up: np.ndarray
    ramp_down: np.ndarray
    redispatchable_mask: np.ndarray
    gen_detached: np.ndarray

    @classmethod
    def from_results(
        cls,
        storage_result: Optional[StorageResult],
        curtail_result: Optional[CurtailmentResult],
        detach_result: Optional[DetachmentResult],
        new_p: np.ndarray,
        state: RedispatchState,
        env: "BaseEnv",
    ) -> "RedispatchConstraints":
        cls_env = type(env)
        gen_detached = env._backend_action.get_gen_detached()
        gen_participating = (
            (new_p > 0.0)
            | (np.abs(state.actual_dispatch) >= 1e-7)
            | (state.target_dispatch != state.actual_dispatch)
        )
        gen_participating[~cls_env.gen_redispatchable] = False
        if cls_env.detachment_is_allowed:
            gen_participating[gen_detached] = False
        return cls(
            new_p=new_p,
            gen_activeprod_t=state.gen_activeprod_t,
            target_dispatch=state.target_dispatch,
            gen_participating=gen_participating,
            amount_storage_mw=dt_float(state.amount_storage),
            sum_curtailment_mw=dt_float(state.sum_curtailment_mw),
            detached_mw=dt_float(state.detached_elements_mw),
            pmin=cls_env.gen_pmin,
            pmax=cls_env.gen_pmax,
            ramp_up=cls_env.gen_max_ramp_up,
            ramp_down=cls_env.gen_max_ramp_down,
            redispatchable_mask=cls_env.gen_redispatchable,
            gen_detached=gen_detached,
        )
