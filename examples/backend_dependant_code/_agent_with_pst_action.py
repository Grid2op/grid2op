# Copyright (c) 2025, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

from grid2op.Agent import BaseAgent
from grid2op.Action import ActionSpace
from grid2op.Environment import BaseEnv
# NB: this agent is, from grid2o point of view
# "cheating" because it can access the env.backend._grid
# attribute when acting (so it does access private information).
# This is for a now a "hack" and will be better handled in
# future grid2op versions.
# This means that, for now, you cannot use properly with "simulate" 
# for example

# NB: as PST are not handled (yet, any contribution welcomed) in grid2op
# this agent is for now limited to environment using pandapower backend.
from pandapower import pandapowerNet


class AgentRandomPST(BaseAgent):
    """
    This agent will perform random action on PST (Phase Shifting Transformer).
    
    You can use it with grid2op >= 1.12.1
    """
    def __init__(self,
                 action_space: ActionSpace):
        super().__init__(action_space)
    
    def act(self, observation, reward, done = False):
        act = super().act(observation, reward, done)
        
        def callback(grid: pandapowerNet):
            which_trafo = self.space_prng.randint(grid.trafo["tap_phase_shifter"].sum())
            trafo_pst_ids = grid.trafo["tap_phase_shifter"].values.nonzero()[0]
            trafo_id = trafo_pst_ids[which_trafo]
            which_tap = self.space_prng.choice([-2, -1, 0, 1, 2], size=1)[0]
            grid.trafo.loc[trafo_id, "tap_pos"] = which_tap
        
        act.backend_dependant_callback = callback
        return act
    
    
# DEPRECATED
class AgentRandomPST_DEPRECATED(BaseAgent):
    """This is a way to do it if you don't have grid2op 1.12.1 or later.
    We recommend you to upgrade grid2op if that is the case
    """
    def __init__(self,
                 action_space: ActionSpace,
                 env: BaseEnv):
        super().__init__(action_space)
        self._backend = env.backend._grid
    
    def act(self, observation, reward, done = False):
        
        # perform a "random" PST action
        which_trafo = self.space_prng.randint(self._backend._grid.trafo["tap_phase_shifter"].sum())
        trafo_pst_ids = self._backend._grid.trafo["tap_phase_shifter"].values.nonzero()[0]
        trafo_id = trafo_pst_ids[which_trafo]
        
        which_tap = self.space_prng.choice([-2, -1, 0, 1, 2], size=1)[0]
        self._backend._grid.trafo.loc[trafo_id, "tap_pos"] = which_tap
        # return the base grid2op action (do nothing in this case)
        return super().act(observation, reward, done)
