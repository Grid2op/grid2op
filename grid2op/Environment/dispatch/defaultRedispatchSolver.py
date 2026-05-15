# Copyright (c) 2026, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

from __future__ import annotations

from typing import List, Optional

import numpy as np
from scipy.optimize import LinearConstraint, minimize

from grid2op.Action import BaseAction
from grid2op.Exceptions import (
    GeneratorTurnedOffTooSoon,
    GeneratorTurnedOnTooSoon,
    IllegalRedispatching,
    ImpossibleRedispatching,
)

from .baseRedispatchSolver import BaseRedispatchSolver
from .dispatchTypes import RedispatchConstraints, RedispatchState


class DefaultRedispatchSolver(BaseRedispatchSolver):
    def reset(self, state: RedispatchState) -> None:
        state.target_dispatch[:] = 0.0
        state.already_modified_gen[:] = False
        state.actual_dispatch[:] = 0.0
        state.gen_uptime[:] = 0
        state.gen_downtime[:] = 0
        state.gen_activeprod_t[:] = 0.0
        state.gen_activeprod_t_redisp[:] = 0.0

    def _update_target_dispatch(self, action: BaseAction, state: RedispatchState):
        if not action._modif_redispatch:
            return state.already_modified_gen

        redisp_act_orig = action._redispatch
        is_redisped = np.abs(redisp_act_orig) > 1e-7
        state.target_dispatch[state.already_modified_gen] += redisp_act_orig[
            state.already_modified_gen
        ]
        first_modified = (~state.already_modified_gen) & is_redisped
        state.target_dispatch[first_modified] = (
            state.actual_dispatch[first_modified] + redisp_act_orig[first_modified]
        )
        state.already_modified_gen[is_redisped] = True
        return state.already_modified_gen

    def _validate_dispatch(
        self,
        action: BaseAction,
        new_p,
        already_modified_gen,
        state: RedispatchState,
    ):
        cls = type(self.env)
        except_ = None
        info_: List[dict] = []
        valid = True

        if action._modif_redispatch:
            redisp_act_orig = action._redispatch.copy()
        else:
            redisp_act_orig = None

        if (
            (redisp_act_orig is not None and (np.abs(redisp_act_orig) <= 1e-7).all())
            and (np.abs(state.target_dispatch) <= 1e-7).all()
            and (np.abs(state.actual_dispatch) <= 1e-7).all()
        ):
            return valid, except_, info_

        if redisp_act_orig is None:
            redisp_act_orig = type(action)._build_attr("_redispatch")

        if (state.target_dispatch > cls.gen_pmax - cls.gen_pmin).any():
            cond_invalid = state.target_dispatch > cls.gen_pmax - cls.gen_pmin
            except_ = IllegalRedispatching(
                "You cannot ask for a dispatch higher than pmax - pmin  [it would be always "
                "invalid because, even if the sepoint is pmin, this dispatch would set it "
                "to a number higher than pmax, which is impossible]. Invalid dispatch for "
                "generator(s): {}".format((cond_invalid).nonzero()[0])
            )
            state.target_dispatch -= redisp_act_orig
            return valid, except_, info_
        if (state.target_dispatch < cls.gen_pmin - cls.gen_pmax).any():
            cond_invalid = state.target_dispatch < cls.gen_pmin - cls.gen_pmax
            except_ = IllegalRedispatching(
                "You cannot ask for a dispatch lower than pmin - pmax  [it would be always "
                "invalid because, even if the sepoint is pmax, this dispatch would set it "
                "to a number bellow pmin, which is impossible]. Invalid dispatch for "
                "generator(s): {}".format((cond_invalid).nonzero()[0])
            )
            state.target_dispatch -= redisp_act_orig
            return valid, except_, info_

        if (redisp_act_orig[np.abs(new_p) <= 1e-7]).any() and self.env._forbid_dispatch_off:
            except_ = IllegalRedispatching("Impossible to dispatch a turned off generator")
            state.target_dispatch -= redisp_act_orig
            return valid, except_, info_

        if self.env._forbid_dispatch_off:
            redisp_act_orig_cut = redisp_act_orig.copy()
            redisp_act_orig_cut[np.abs(new_p) <= 1e-7] = 0.0
            if (redisp_act_orig_cut != redisp_act_orig).any():
                info_.append(
                    {
                        "INFO: redispatching cut because generator will be turned_off": (
                            redisp_act_orig_cut != redisp_act_orig
                        ).nonzero()[0]
                    }
                )
        return valid, except_, info_

    def _detect_infeasible_dispatch(
        self,
        constraints: RedispatchConstraints,
        incr_in_chronics,
        avail_down,
        avail_up,
        state: RedispatchState,
    ):
        except_ = None
        sum_move = (
            incr_in_chronics.sum()
            + constraints.amount_storage_mw
            - constraints.sum_curtailment_mw
            + constraints.detached_mw
        )
        avail_down_sum = avail_down.sum()
        avail_up_sum = avail_up.sum()
        gen_setpoint = state.gen_activeprod_t_redisp[self.env.gen_redispatchable]
        if sum_move > avail_up_sum:
            msg = self.env.DETAILED_REDISP_ERR_MSG.format(
                sum_move=sum_move,
                avail_up_sum=avail_up_sum,
                gen_setpoint=np.round(gen_setpoint, decimals=2),
                ramp_up=self.env.gen_max_ramp_up[self.env.gen_redispatchable],
                gen_pmax=self.env.gen_pmax[self.env.gen_redispatchable],
                avail_up=np.round(avail_up, decimals=2),
                increase="increase",
                decrease="decrease",
                maximum="maximum",
                pmax="pmax",
                max_ramp_up="max_ramp_up",
            )
            except_ = ImpossibleRedispatching(msg)
        elif sum_move < avail_down_sum:
            msg = self.env.DETAILED_REDISP_ERR_MSG.format(
                sum_move=sum_move,
                avail_up_sum=avail_down_sum,
                gen_setpoint=np.round(gen_setpoint, decimals=2),
                ramp_up=self.env.gen_max_ramp_down[self.env.gen_redispatchable],
                gen_pmax=self.env.gen_pmin[self.env.gen_redispatchable],
                avail_up=np.round(avail_up, decimals=2),
                increase="decrease",
                decrease="increase",
                maximum="minimum",
                pmax="pmin",
                max_ramp_up="max_ramp_down",
            )
            except_ = ImpossibleRedispatching(msg)
        return except_

    def _solve(self, constraints: RedispatchConstraints, state: RedispatchState):
        except_ = None
        cls = type(self.env)
        this_dt_float = float
        new_p = constraints.new_p
        if self.env.nb_time_step == 0:
            state.gen_activeprod_t_redisp[:] = new_p

        gen_participating = constraints.gen_participating.copy()
        incr_in_chronics = new_p - (state.gen_activeprod_t_redisp - state.actual_dispatch)

        p_min_down = cls.gen_pmin[gen_participating] - state.gen_activeprod_t_redisp[gen_participating]
        avail_down = np.maximum(p_min_down, -cls.gen_max_ramp_down[gen_participating])
        p_max_up = cls.gen_pmax[gen_participating] - state.gen_activeprod_t_redisp[gen_participating]
        avail_up = np.minimum(p_max_up, cls.gen_max_ramp_up[gen_participating])
        except_ = self._detect_infeasible_dispatch(
            constraints,
            incr_in_chronics[gen_participating],
            avail_down,
            avail_up,
            state,
        )
        if except_ is not None:
            if (
                self.env._parameters.IGNORE_MIN_UP_DOWN_TIME
                and self.env._parameters.ALLOW_DISPATCH_GEN_SWITCH_OFF
            ):
                gen_participating_tmp = self.env.gen_redispatchable.copy()
                if cls.detachment_is_allowed:
                    gen_participating_tmp[constraints.gen_detached] = False
                p_min_down_tmp = (
                    cls.gen_pmin[gen_participating_tmp]
                    - state.gen_activeprod_t_redisp[gen_participating_tmp]
                )
                avail_down_tmp = np.maximum(
                    p_min_down_tmp, -cls.gen_max_ramp_down[gen_participating_tmp]
                )
                p_max_up_tmp = (
                    cls.gen_pmax[gen_participating_tmp]
                    - state.gen_activeprod_t_redisp[gen_participating_tmp]
                )
                avail_up_tmp = np.minimum(
                    p_max_up_tmp, cls.gen_max_ramp_up[gen_participating_tmp]
                )
                except_tmp = self._detect_infeasible_dispatch(
                    constraints,
                    incr_in_chronics[gen_participating_tmp],
                    avail_down_tmp,
                    avail_up_tmp,
                    state,
                )
                if except_tmp is None:
                    gen_participating = gen_participating_tmp
                    except_ = None
                else:
                    return except_tmp
            else:
                return except_

        target_vals = state.target_dispatch[gen_participating] - state.actual_dispatch[gen_participating]
        already_modified_gen_me = state.already_modified_gen[gen_participating]
        target_vals_me = target_vals[already_modified_gen_me]
        nb_dispatchable = gen_participating.sum()
        tmp_zeros = np.zeros((1, nb_dispatchable), dtype=this_dt_float)
        coeffs = 1.0 / (self.env.gen_max_ramp_up + self.env.gen_max_ramp_down + self.env._epsilon_poly)
        weights = np.ones(nb_dispatchable) * coeffs[gen_participating]
        weights /= weights.sum()

        if target_vals_me.shape[0] == 0:
            already_modified_gen_me[:] = True
            target_vals_me = target_vals[already_modified_gen_me]

        scale_x = max(np.max(np.abs(state.actual_dispatch)), 1.0)
        scale_x = this_dt_float(scale_x)
        target_vals_me_optim = 1.0 * (target_vals_me / scale_x)
        target_vals_me_optim = target_vals_me_optim.astype(this_dt_float)

        scale_objective = max(0.5 * np.abs(target_vals_me_optim).sum() ** 2, 1.0)
        scale_objective = np.round(scale_objective, decimals=4)
        scale_objective = this_dt_float(scale_objective)

        mat_sum_0_no_turn_on = np.ones((1, nb_dispatchable), dtype=this_dt_float)
        const_sum_0_no_turn_on = (
            np.zeros(1, dtype=this_dt_float)
            + constraints.amount_storage_mw
            - constraints.sum_curtailment_mw
            + constraints.detached_mw
        )

        new_p_th = new_p[gen_participating] + state.actual_dispatch[gen_participating]
        p_min_const = self.env.gen_pmin[gen_participating] - new_p_th
        ramp_down_const = -self.env.gen_max_ramp_down[gen_participating] - incr_in_chronics[gen_participating]
        min_disp = np.maximum(p_min_const, ramp_down_const).astype(this_dt_float)

        p_max_const = self.env.gen_pmax[gen_participating] - new_p_th
        ramp_up_const = self.env.gen_max_ramp_up[gen_participating] - incr_in_chronics[gen_participating]
        max_disp = np.minimum(p_max_const, ramp_up_const).astype(this_dt_float)

        added = 0.5 * self.env._epsilon_poly
        equality_const = LinearConstraint(
            mat_sum_0_no_turn_on,
            const_sum_0_no_turn_on / scale_x,
            const_sum_0_no_turn_on / scale_x,
        )
        mat_pmin_max_ramps = np.eye(nb_dispatchable)
        ineq_const = LinearConstraint(
            mat_pmin_max_ramps,
            (min_disp - added) / scale_x,
            (max_disp + added) / scale_x,
        )

        x0 = np.zeros(gen_participating.sum(), dtype=this_dt_float)
        if (np.abs(state.target_dispatch) >= 1e-7).any() or state.already_modified_gen.any():
            gen_for_x0 = np.abs(state.target_dispatch[gen_participating]) >= 1e-7
            gen_for_x0 |= state.already_modified_gen[gen_participating]
            x0[gen_for_x0] = (
                state.target_dispatch[gen_participating][gen_for_x0]
                - state.actual_dispatch[gen_participating][gen_for_x0]
            ) / scale_x
            can_adjust = np.abs(x0) <= 1e-7
            if can_adjust.any():
                init_sum = x0.sum()
                denom_adjust = (1.0 / weights[can_adjust]).sum()
                if denom_adjust <= 1e-2:
                    denom_adjust = 1.0
                x0[can_adjust] = -init_sum / (weights[can_adjust] * denom_adjust)
        else:
            x0 -= state.actual_dispatch[gen_participating] / scale_x

        def target(actual_dispatchable):
            quad_ = (
                actual_dispatchable[already_modified_gen_me] - target_vals_me_optim
            ) ** 2
            coeffs_quads = weights[already_modified_gen_me] * quad_
            coeffs_quads_const = coeffs_quads.sum()
            coeffs_quads_const /= scale_objective
            return coeffs_quads_const

        def jac(actual_dispatchable):
            res_jac = 1.0 * tmp_zeros
            res_jac[0, already_modified_gen_me] = (
                2.0
                * weights[already_modified_gen_me]
                * (actual_dispatchable[already_modified_gen_me] - target_vals_me_optim)
            )
            res_jac /= scale_objective
            return res_jac.reshape(-1)

        res = minimize(
            target,
            x0,
            method="SLSQP",
            constraints=[equality_const, ineq_const],
            options={
                "eps": max(this_dt_float(self.env._epsilon_poly / scale_x), 1e-6),
                "ftol": max(this_dt_float(self.env._epsilon_poly / scale_x), 1e-6),
                "disp": False,
            },
            jac=jac,
        )
        if res.success:
            state.actual_dispatch[gen_participating] += res.x * scale_x
        else:
            mat_const = np.concatenate((mat_sum_0_no_turn_on, mat_pmin_max_ramps))
            downs = np.concatenate(
                (const_sum_0_no_turn_on / scale_x, (min_disp - added) / scale_x)
            )
            ups = np.concatenate(
                (const_sum_0_no_turn_on / scale_x, (max_disp + added) / scale_x)
            )
            vals = np.matmul(mat_const, res.x)
            ok_down = np.all(vals - downs >= -self.env._tol_poly)
            ok_up = np.all(vals - ups <= self.env._tol_poly)
            if ok_up and ok_down:
                state.actual_dispatch[gen_participating] += res.x * scale_x
            else:
                error_dispatch = (
                    "Redispatching automaton terminated with error (no more information available "
                    'at this point):\n"{}"'.format(res.message)
                )
                except_ = ImpossibleRedispatching(error_dispatch)
        return except_

    def solve(
        self,
        constraints: RedispatchConstraints,
        state: RedispatchState,
    ) -> Optional[Exception]:
        if not self.env._parameters.ENV_DOES_REDISPATCHING:
            state.actual_dispatch[:] = state.target_dispatch.copy()
            return None
        mismatch = np.abs(state.actual_dispatch - state.target_dispatch)
        if (
            np.abs((state.actual_dispatch).sum()) >= self.env._tol_poly
            or np.max(mismatch) >= self.env._tol_poly
            or np.abs(state.amount_storage) >= self.env._tol_poly
            or np.abs(state.sum_curtailment_mw) >= self.env._tol_poly
            or np.abs(state.detached_elements_mw) >= self.env._tol_poly
        ):
            return self._solve(constraints, state)
        return None

    def _check_updown_times(self, gen_up_before, redisp_act, state: RedispatchState):
        except_ = None
        cls = type(self.env)
        gen_up_after = state.gen_activeprod_t.copy()
        if "prod_p" in self.env._env_modification._dict_inj:
            tmp = self.env._env_modification._dict_inj["prod_p"]
            indx_ok = np.isfinite(tmp)
            gen_up_after[indx_ok] = self.env._env_modification._dict_inj["prod_p"][indx_ok]
        gen_up_after += redisp_act
        gen_up_after = np.abs(gen_up_after) > 1e-7

        gen_disconnected_this = gen_up_before & (~gen_up_after)
        gen_connected_this_timestep = (~gen_up_before) & gen_up_after & ~self.env._gens_detached
        gen_still_connected = gen_up_before & gen_up_after
        gen_still_disconnected = ((~gen_up_before) & (~gen_up_after)) | self.env._gens_detached
        if (
            not self.env._ignore_min_up_down_times
            and (
                state.gen_downtime[gen_connected_this_timestep]
                < cls.gen_min_downtime[gen_connected_this_timestep]
            ).any()
        ):
            id_gen = (
                state.gen_downtime[gen_connected_this_timestep]
                < cls.gen_min_downtime[gen_connected_this_timestep]
            )
            id_gen = (id_gen).nonzero()[0]
            id_gen = (gen_connected_this_timestep[id_gen]).nonzero()[0]
            except_ = GeneratorTurnedOnTooSoon(
                "Some generator has been connected too early ({})".format(id_gen)
            )
            return except_
        else:
            state.gen_downtime[gen_connected_this_timestep] = -1
            state.gen_uptime[gen_connected_this_timestep] = 0

        if (
            not self.env._ignore_min_up_down_times
            and (
                state.gen_uptime[gen_disconnected_this]
                < cls.gen_min_uptime[gen_disconnected_this]
            ).any()
        ):
            id_gen = (
                state.gen_uptime[gen_disconnected_this]
                < cls.gen_min_uptime[gen_disconnected_this]
            )
            id_gen = (id_gen).nonzero()[0]
            id_gen = (gen_disconnected_this[id_gen]).nonzero()[0]
            except_ = GeneratorTurnedOffTooSoon(
                "Some generator has been disconnected too early ({})".format(id_gen)
            )
            return except_
        else:
            state.gen_downtime[gen_disconnected_this] = 0
            state.gen_uptime[gen_disconnected_this] = -1

        state.gen_uptime[gen_still_connected] += 1
        state.gen_downtime[gen_still_disconnected] += 1
        return except_
