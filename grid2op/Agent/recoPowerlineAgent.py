# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

from typing import List
from grid2op.Observation import BaseObservation
from grid2op.Action import BaseAction, ActionSpace

from grid2op.Agent.greedyAgent import GreedyAgent


class RecoPowerlineAgent(GreedyAgent):
    """
    This is a :class:`GreedyAgent` example, which will attempt to reconnect powerlines: for each disconnected powerline
    that can be reconnected, it will simulate the effect of reconnecting it. And reconnect the one that lead to the
    highest simulated reward.

    """

    def __init__(self, action_space: ActionSpace, simulated_time_step : int =1):
        GreedyAgent.__init__(self, action_space, simulated_time_step=simulated_time_step)

    def _get_tested_action(self, observation: BaseObservation) -> List[BaseAction]:
        res = [self.action_space({})]  # add the do nothing
        line_stat_s = observation.line_status
        cooldown = observation.time_before_cooldown_line
        can_be_reco = ~line_stat_s & (cooldown == 0)
        if can_be_reco.any():
            res = [
                self.action_space({"set_line_status": [(id_, +1)]})
                for id_ in (can_be_reco).nonzero()[0]
            ]
        return res
