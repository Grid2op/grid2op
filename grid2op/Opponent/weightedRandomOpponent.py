# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import warnings
import numpy as np
import copy
from typing import TYPE_CHECKING, Optional

from grid2op.dtypes import dt_int
from grid2op.Opponent.baseOpponent  import BaseOpponent
from grid2op.Exceptions import OpponentError

if TYPE_CHECKING:
    from grid2op.Action import BaseAction
    from grid2op.Observation import BaseObservation


class WeightedRandomOpponent(BaseOpponent):
    """
    This opponent will disconnect lines randomly among the attackable lines `lines_attacked`.
    The sampling is weighted by the lines current usage rate divided by some factor `rho_normalization`
    (see init for more details).

    When an attack becomes possible, the time of the attack will be sampled uniformly
    in the next `attack_period` steps (see init).

    With `independent_timing=True`, the attack times are drawn from a random generator of their own
    (seeded from the opponent's generator at each reset), separate from the one that chooses the line.
    Two opponents built on this class that differ only in how they choose the line then attack at the
    same steps of a seeded episode (as long as each of them still has a connected attackable line when its
    attack time comes), which makes paired comparisons of opponents possible (see :class:`LookaheadLineOpponent`).
    """

    def __init__(self, action_space):
        BaseOpponent.__init__(self, action_space)
        self._do_nothing = None
        self._attacks = None
        self._lines_ids = None
        self._next_attack_time = None
        self._attack_period = None
        self._rho_normalization = None
        self._independent_timing = False
        self._timing_prng = None

        # this is the constructor:
        # it should have the exact same signature as here

    def init(
        self,
        partial_env,
        lines_attacked=[],
        rho_normalization=[],
        attack_period=12 * 24,
        independent_timing=False,
        **kwargs,
    ):
        """
        Generic function used to initialize the derived classes. For example, if an opponent reads from a file, the
        path where is the file is located should be pass with this method.

        Parameters
        ----------
        lines_attacked: ``list``
            The list of lines that the WeightedRandomOpponent should be able to disconnect

        rho_normalization: ``list``
            The list of mean usage rates for the attackable lines. Should have
            the same length as lines_attacked. If no value is given, no normalization will be performed.
            The weights for sampling the attacked line are rho / rho_normalization.

        attack_period: ``int``
            The number of steps among which the attack may happen.
            If attack_period=10, then whenever an attack can be made, it will happen in the 10
            next steps.

        independent_timing: ``bool``
            If ``True``, the attack times are drawn from a dedicated random generator, so that they
            do not depend on how the line is chosen. Default ``False`` (one generator for both, as before).
        """

        self._set_line_id(lines_attacked)

        # Pre-build attacks actions
        self._prebuild_action()

        # Usage rates normalization
        self._rho_normalization = np.ones_like(lines_attacked)
        if len(rho_normalization) == 0:
            warnings.warn(
                "The usage rate normalization is not specified. No normalization will be performed."
            )
        elif len(rho_normalization) != len(lines_attacked):
            raise Warning(
                "The usage rate normalization must have the same length as the number "
                "of attacked lines. No normalization will be performed."
            )
        else:
            self._rho_normalization = np.array(rho_normalization)

        # Opponent's attack period
        self._attack_period = attack_period
        if self._attack_period <= 0:
            raise OpponentError("Opponent attack cooldown need to be > 0")
        self._independent_timing = bool(independent_timing)

    def reset(self, initial_budget):
        self._next_attack_time = None
        if self._independent_timing:
            self._timing_prng = self._make_timing_prng()

    def _make_timing_prng(self):
        return np.random.RandomState(self.space_prng.randint(np.iinfo(dt_int).max))

    def tell_attack_continues(self, observation, agent_action, env_action, budget):
        self._next_attack_time = None

    def attack(self, observation, agent_action, env_action, budget, previous_fails):
        """
        This method is the equivalent of "attack" for a regular agent.

        Opponent, in this framework can have more information than a regular agent (in particular it can
        view time step t+1), it has access to its current budget etc.

        Parameters
        ----------
        observation: :class:`grid2op.Observation.Observation`
            The last observation (at time t)

        opp_reward: ``float``
            THe opponent "reward" (equivalent to the agent reward, but for the opponent) TODO do i add it back ???

        done: ``bool``
            Whether the game ended or not TODO do i add it back ???

        agent_action: :class:`grid2op.Action.Action`
            The action that the agent took

        env_action: :class:`grid2op.Action.Action`
            The modification that the environment will take.

        budget: ``float``
            The current remaining budget (if an action is above this budget, it will be replaced by a do nothing.

        previous_fails: ``bool``
            Wheter the previous attack failed (due to budget or ambiguous action)

        Returns
        -------
        attack: :class:`grid2op.Action.Action`
            The attack performed by the opponent. In this case, a do nothing, all the time.

        duration: ``int``
            The duration of the attack
        """
        # TODO maybe have a class "GymOpponent" where the observation would include the budget  and all other
        # TODO information, and forward something to the "act" method.

        # During creation of the environment, do not attack
        if observation is None:
            return None, 0

        # Decide the time of the next attack
        if self._next_attack_time is None:
            if self._independent_timing and self._timing_prng is None:
                self._timing_prng = self._make_timing_prng()
            timing_prng = self._timing_prng if self._independent_timing else self.space_prng
            self._next_attack_time = 1 + timing_prng.randint(self._attack_period)
        self._next_attack_time -= 1

        # If the attack time has not come yet, do not attack
        if self._next_attack_time > 0:
            return None, 0

        # If all attackable lines are disconnected, do not attack
        status = observation.line_status[self._lines_ids]
        if not status.sum():
            return None, 0

        attack = self._choose_attack(observation, status)
        if attack is None:
            return None, 0
        return attack, None

    def _choose_attack(self, observation: "BaseObservation", status: np.ndarray) -> Optional["BaseAction"]:
        """
        Choose the attack among the connected attackable lines (`status` is ``True`` for them),
        or return ``None`` to not attack. Here: sampled with probability proportional to the usage rate.
        """
        available_attacks = self._attacks[status]
        rho = observation.rho[self._lines_ids][status] / self._rho_normalization[status]
        rho_sum = rho.sum()
        if rho_sum <= 0.0:
            # this case can happen if a powerline has a flow of 0.0 but is connected, and it's the only one
            # that can be attacked... Pretty rare hey !
            return None
        return self.space_prng.choice(available_attacks, p=rho / rho_sum)

    def _custom_deepcopy_for_copy(self, new_obj, dict_=None):
        super()._custom_deepcopy_for_copy(new_obj, dict_)
        if dict_ is None:
            dict_ = {}

        new_obj._attacks = copy.deepcopy(self._attacks)
        new_obj._lines_ids = copy.deepcopy(self._lines_ids)
        new_obj._next_attack_time = copy.deepcopy(self._next_attack_time)
        new_obj._attack_period = copy.deepcopy(self._attack_period)
        new_obj._rho_normalization = copy.deepcopy(self._rho_normalization)
        new_obj._independent_timing = self._independent_timing
        new_obj._timing_prng = copy.deepcopy(self._timing_prng)
