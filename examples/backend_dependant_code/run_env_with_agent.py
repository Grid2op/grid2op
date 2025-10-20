# Copyright (c) 2025, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import numpy as np
import grid2op
from _obs_with_n1 import ObsWithN1
from _reward_n1 import N1Reward
from _agent_with_pst_action import AgentRandomPST

# simulate contingency on the first 5 lines (why not ?)
li_lines = np.arange(5) 
# env where PST will be modified
env = grid2op.make("l2rpn_idf_2023",
                   test=True, 
                   observation_class=ObsWithN1,
                   kwargs_observation={"n1_li": 1 * li_lines},
                   reward_class=N1Reward(n1_li= 1 * li_lines))

# small hack to pretend there are PST in the grid
# this is totally useless if the grid already has PST
for grid in [env.backend._grid,
             env.backend._PandaPowerBackend__pp_backend_initial_grid]: 
    add_PSTs = [0, 1]
    grid.trafo.loc[add_PSTs, "tap_step_degree"] = 1.
    grid.trafo.loc[add_PSTs, "tap_step_percent"] = 0.
    grid.trafo.loc[add_PSTs, "tap_phase_shifter"] = True
    grid.trafo.loc[add_PSTs, "tap_pos"] = 0
    
# regular env without PST modification (as a reference)
env_without_pst = env.copy()

pst_agent = AgentRandomPST(action_space=env.action_space)
pst_agent.seed(0)

# initial state
obs = env.reset(seed=0, options={"time serie id": 0})
# legacy: following line used to be required for grid2op 1.12.1 but is not anymore
# pst_agent._backend = env.backend  # do not forget to synch the agent with the env

obs_without_pst = env_without_pst.reset(seed=0, options={"time serie id": 0})
# check there is no difference
assert (obs.rho - obs_without_pst.rho).max() <= 1e-5

# perform a (random) PST action on one, check effect 
# by comparison with the baseline env
pst_action = pst_agent.act(obs, None, None)
# for grid2op, the pst_action is equivalent to the
# "do nothing" action, grid2op has (for now !, this will be improved
# in next release) no way of knowing an "action" took place.
# because the agent action modified the env directly
next_obs, reward, *_ = env.step(pst_action)
next_obs_without_pst, reward_without_pst, *_ = env_without_pst.step(pst_action)
print("Maximum difference (in relative flows) with / without the action: "
      f"{(next_obs.rho - next_obs_without_pst.rho).max() * 100.:.2f} % of thermal limit")
print("Difference (in reward flows) with / without the action: "
      f"{reward - reward_without_pst}")

# do another action
pst_action = pst_agent.act(obs, None, None)
next_obs,reward, * _ = env.step(pst_action)
next_obs_without_pst, reward_without_pst, *_ = env_without_pst.step(pst_action)
print("Maximum difference (in relative flows) with / without the action: "
      f"{(next_obs.rho - next_obs_without_pst.rho).max() * 100.:.2f} % of thermal limit")
print("Difference (in reward flows) with / without the action: "
      f"{reward - reward_without_pst}")

# do another action
pst_action = pst_agent.act(obs, None, None)
next_obs, reward, * _ = env.step(pst_action)
next_obs_without_pst, reward_without_pst, *_ = env_without_pst.step(pst_action)
print("Maximum difference (in relative flows) with / without the action: "
      f"{(next_obs.rho - next_obs_without_pst.rho).max() * 100.:.2f} % of thermal limit")
print("Difference (in reward flows) with / without the action: "
      f"{reward - reward_without_pst}")
