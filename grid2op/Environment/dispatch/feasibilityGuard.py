from __future__ import annotations

import numpy as np

from grid2op.dtypes import dt_float

from .dispatchTypes import CurtailmentResult, GuardInfo, RedispatchState, StorageResult


class FeasibilityGuard:
    def __init__(self, env) -> None:
        self.env = env

    def _compute_ramp_budget(self, new_p, state: RedispatchState):
        th_max = np.minimum(
            state.gen_activeprod_t_redisp[self.env.gen_redispatchable]
            + self.env.gen_max_ramp_up[self.env.gen_redispatchable],
            self.env.gen_pmax[self.env.gen_redispatchable],
        )
        th_min = np.maximum(
            state.gen_activeprod_t_redisp[self.env.gen_redispatchable]
            - self.env.gen_max_ramp_down[self.env.gen_redispatchable],
            self.env.gen_pmin[self.env.gen_redispatchable],
        )

        max_total_up = (th_max - new_p[self.env.gen_redispatchable]).sum()
        max_total_down = (th_min - new_p[self.env.gen_redispatchable]).sum()
        return max_total_down, max_total_up

    def _readjust_curtailment(
        self,
        total_curtailment,
        new_p_th,
        new_p,
        state: RedispatchState,
    ) -> None:
        state.sum_curtailment_mw += total_curtailment
        state.sum_curtailment_mw_prev += total_curtailment
        if total_curtailment > self.env._tol_poly:
            curtailed = new_p_th - new_p
        else:
            new_p_with_previous_curtailment = 1.0 * new_p_th
            self.env._curtailment_module._apply_limits(
                new_p_with_previous_curtailment, state.limit_curtailment_prev
            )
            curtailed = new_p_th - new_p_with_previous_curtailment

        curt_sum = curtailed.sum()
        if abs(curt_sum) > self.env._tol_poly:
            curtailed[~self.env.gen_renewable] = 0.0
            curtailed *= total_curtailment / curt_sum
            new_p[self.env.gen_renewable] += curtailed[self.env.gen_renewable]

    def _readjust_storage(self, total_storage, state: RedispatchState) -> None:
        new_act_storage = 1.0 * state.storage_power
        sum_this_step = new_act_storage.sum()
        if abs(total_storage) < abs(sum_this_step):
            modif_storage = new_act_storage * total_storage / sum_this_step
        else:
            new_act_storage = 1.0 * state.storage_power_prev
            sum_this_step = new_act_storage.sum()
            if abs(sum_this_step) > 1e-1:
                modif_storage = new_act_storage * total_storage / sum_this_step
            else:
                modif_storage = new_act_storage

        coeff_p_to_E = self.env.delta_time_seconds / 3600.0
        state.storage_power -= modif_storage

        is_discharging = state.storage_power < 0.0
        is_charging = state.storage_power > 0.0
        modif_storage[is_discharging] /= type(self.env).storage_discharging_efficiency[
            is_discharging
        ]
        modif_storage[is_charging] *= type(self.env).storage_charging_efficiency[
            is_charging
        ]

        state.storage_current_charge -= coeff_p_to_E * modif_storage
        state.amount_storage -= total_storage
        state.amount_storage_prev -= total_storage

    def check_and_clamp(
        self,
        storage_result: StorageResult,
        curtail_result: CurtailmentResult,
        new_p,
        new_p_th,
        state: RedispatchState,
    ) -> GuardInfo:
        gen_redisp = self.env.gen_redispatchable
        normal_increase = new_p - (
            state.gen_activeprod_t_redisp - state.actual_dispatch
        )
        normal_increase = normal_increase[gen_redisp]
        p_min_down = self.env.gen_pmin[gen_redisp] - state.gen_activeprod_t_redisp[gen_redisp]
        avail_down = np.maximum(p_min_down, -self.env.gen_max_ramp_down[gen_redisp])
        p_max_up = self.env.gen_pmax[gen_redisp] - state.gen_activeprod_t_redisp[gen_redisp]
        avail_up = np.minimum(p_max_up, self.env.gen_max_ramp_up[gen_redisp])

        sum_move = normal_increase.sum() + state.amount_storage - state.sum_curtailment_mw
        total_storage_curtail = state.amount_storage - state.sum_curtailment_mw
        update_env_act = False
        total_curtailment = dt_float(0.0)
        total_storage = dt_float(0.0)

        if abs(total_storage_curtail) >= self.env._tol_poly:
            too_much = 0.0
            if sum_move > avail_up.sum():
                too_much = dt_float(sum_move - avail_up.sum() + self.env._tol_poly)
                state.limited_before = too_much
            elif sum_move < avail_down.sum():
                too_much = dt_float(sum_move - avail_down.sum() - self.env._tol_poly)
                state.limited_before = too_much
            elif np.abs(state.limited_before) >= self.env._tol_poly:
                update_env_act = True
                too_much = min(avail_up.sum() - self.env._tol_poly, state.limited_before)
                state.limited_before -= too_much
                too_much = state.limited_before

            if abs(too_much) > self.env._tol_poly:
                total_curtailment = dt_float(
                    -state.sum_curtailment_mw / total_storage_curtail * too_much
                )
                total_storage = dt_float(
                    state.amount_storage / total_storage_curtail * too_much
                )
                update_env_act = True

                if np.sign(total_curtailment) != np.sign(total_storage):
                    total_curtailment = (
                        too_much
                        if np.sign(total_curtailment) == np.sign(too_much)
                        else 0.0
                    )
                    total_storage = (
                        too_much if np.sign(total_storage) == np.sign(too_much) else 0.0
                    )

                self._readjust_curtailment(total_curtailment, new_p_th, new_p, state)
                self._readjust_storage(total_storage, state)

            if update_env_act:
                self.env._curtailment_module._update_env_action(new_p)

        return GuardInfo(
            total_curtailment=dt_float(total_curtailment),
            total_storage=dt_float(total_storage),
            updated_env_action=update_env_act,
        )
