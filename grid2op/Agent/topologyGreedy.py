# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

from typing import List, Optional
from grid2op.Observation import BaseObservation
from grid2op.Action import BaseAction, ActionSpace
from grid2op.Agent.greedyAgent import GreedyAgent


class TopologyGreedy(GreedyAgent):
    """
    This is a :class:`GreedyAgent` example, which will attempt to reconfigure the substations connectivity.

    It will choose among:

      - doing nothing
      - changing the topology of one substation.

    To choose, it will simulate the outcome of all actions, and then chose the action leading to the best rewards.

    """

    def __init__(self, action_space: ActionSpace, simulated_time_step : int =1):
        GreedyAgent.__init__(self, action_space, simulated_time_step=simulated_time_step)
        self.tested_action : Optional[list[BaseAction]] = None

    def _get_tested_action(self, observation: BaseObservation) -> List[BaseAction]:
        if self.tested_action is None:
            res = [self.action_space({})]  # add the do nothing
            # better use "get_all_unitary_topologies_set" and not "get_all_unitary_topologies_change"
            # maybe "change" are still "bugged" (in the sens they don't count all topologies exactly once)
            res += self.action_space.get_all_unitary_topologies_set(self.action_space)
            self.tested_action = res
        return self.tested_action
