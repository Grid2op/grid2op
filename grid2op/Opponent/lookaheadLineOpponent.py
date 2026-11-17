# Copyright (c) 2019-2026, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

from typing import TYPE_CHECKING, Optional

import numpy as np

from grid2op.Exceptions import BaseObservationError, OpponentError
from grid2op.Opponent.weightedRandomOpponent import WeightedRandomOpponent

if TYPE_CHECKING:
    from grid2op.Action import BaseAction
    from grid2op.Observation import BaseObservation


class LookaheadLineOpponent(WeightedRandomOpponent):
    """
    This opponent disconnects, among the attackable lines `lines_attacked`, the connected line whose
    disconnection gives the highest maximum usage rate (``rho``) on the grid state of the current observation.

    Each candidate is evaluated with the simulator of the current observation
    (:func:`grid2op.Observation.BaseObservation.get_simulator`), which runs one power flow per candidate
    without protections or cooldowns. A candidate whose power flow does not converge counts as the
    strongest attack. Candidates are scanned in the order of `lines_attacked`, and a later candidate only
    replaces the current choice if its score is higher by more than ``1e-9``, so near ties go to the line listed
    first. No random number is used to choose the line.

    The time of the attack follows the :class:`WeightedRandomOpponent` rule (uniform in the next
    `attack_period` steps once an attack is possible). The simulator is only used at the step where an
    attack starts, and these power flows are not added to the count of the agent's simulations
    (``env.nb_highres_called``).

    Notes
    -----
    To compare this opponent with a :class:`WeightedRandomOpponent` at the same attack times, pass
    ``independent_timing=True`` in the `kwargs_opponent` of both environments and seed them the same way.
    They then attack at the same steps, as long as each of them still has a connected attackable line
    when its attack time comes.

    The `rho_normalization` argument of :class:`WeightedRandomOpponent` is not used.

    This opponent raises an :class:`grid2op.Exceptions.OpponentError` when the observations have no
    simulator: for example when forecasts are deactivated (``env.deactivate_forecast()``, or a
    :class:`grid2op.gym_compat.GymEnv`, where ``with_forecast=False`` is the default) or when the backend
    cannot be copied.

    Examples
    --------

    .. code-block:: python

        import grid2op
        from grid2op.Action import PowerlineSetAction
        from grid2op.Opponent import LookaheadLineOpponent, BaseActionBudget

        env = grid2op.make("l2rpn_case14_sandbox",
                           opponent_attack_cooldown=12 * 24,
                           opponent_attack_duration=12 * 4,
                           opponent_budget_per_ts=0.17,
                           opponent_init_budget=144.,
                           opponent_action_class=PowerlineSetAction,
                           opponent_class=LookaheadLineOpponent,
                           opponent_budget_class=BaseActionBudget,
                           kwargs_opponent={"lines_attacked": ["1_3_3", "1_4_4", "3_6_15"],
                                            "attack_period": 12 * 24,
                                            "independent_timing": True})
    """

    _TIE_TOL = 1e-9

    def init(self, partial_env, lines_attacked=(), attack_period=12 * 24, independent_timing=False, **kwargs):
        # the usage rate normalization is only used to sample a line, which this opponent does not do
        kwargs.pop("rho_normalization", None)
        super().init(partial_env,
                     lines_attacked=lines_attacked,
                     rho_normalization=[1.0] * len(lines_attacked),
                     attack_period=attack_period,
                     independent_timing=independent_timing,
                     **kwargs)

    def _choose_attack(self, observation: "BaseObservation", status: np.ndarray) -> Optional["BaseAction"]:
        """
        Return the attack on the connected attackable line whose simulated disconnection leads
        to the highest maximum usage rate.
        """
        try:
            simulator = observation.get_simulator()
        except BaseObservationError as exc:
            raise OpponentError("LookaheadLineOpponent needs `obs.get_simulator()`, which is not available here "
                                "(are the forecasts deactivated, or is the backend impossible to copy?)") from exc
        # lazy import to prevent circular references
        from grid2op.Observation.highresSimCounter import HighResSimCounter
        # the opponent's power flows should not count as simulations made by the agent
        simulator._highres_sim_counter = HighResSimCounter()

        best_attack = None
        best_score = -np.inf
        for position in np.flatnonzero(status):
            attack = self._attacks[position]
            result = simulator.predict(act=attack, do_copy=True)
            if not result.converged or result.current_obs is None:
                score = np.inf
            else:
                score = float(np.max(result.current_obs.rho))
            if best_attack is None or score > best_score + type(self)._TIE_TOL:
                best_attack = attack
                best_score = score
        return best_attack
