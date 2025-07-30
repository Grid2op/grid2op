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

# default config
print("Default configuration, all lines disconnected")
env = grid2op.make("l2rpn_case14_sandbox", 
                   observation_class=ObsWithN1,
                   reward_class=N1Reward())

obs = env.reset()
print(f"{obs.n1_vals }")
obs, reward, done, info = env.step(env.action_space())
print(f"{obs.n1_vals }")
print(f"{reward = }")

# with a specific list of n1 to simulate
print("Custom configuration, only line id 0..5 will be disconnected")
li_lines = np.arange(5)
env = grid2op.make("l2rpn_case14_sandbox", 
                   observation_class=ObsWithN1,
                   kwargs_observation={"n1_li": 1 * li_lines},
                   reward_class=N1Reward(n1_li= 1 * li_lines))

obs = env.reset()
print(f"{obs.n1_vals }")
obs, reward, done, info = env.step(env.action_space())
print(f"{obs.n1_vals }")
print(f"{reward = }")

# NB: reward (score) and observation are two different objects
# so you can have different list of simulated n-1 when you
# train the agent and when you score it.

# NB: reward (score) and observation are independant:
# so you can give the agent the "sum" or "max" of all
# flows for each n-1
# but score it with maximum flows for example.

# NB: as of now, no optimization are done in the "N1Reward"
# to reuse the n-1 computation of the Observation (and vice versa)
# this might be time consuming. An easy optimization to perform
# in the reward class, would be to reuse the information in 
# env.get_obs(_do_copy=False).n1_vals