# Copyright (c) 2025, RTE (https://www.rte-france.com)
# See AUTHORS.txt and https://github.com/Grid2Op/grid2op/pull/319
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.


import warnings
import grid2op
import numpy as np
import unittest

from grid2op.Action import CompleteAction
from grid2op.Observation import CompleteObservation


class TestIssue752(unittest.TestCase):
    def setUp(self):
        env_name = "educ_case14_storage"
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make(env_name,
                                    test=True,
                                    observation_class=CompleteObservation,
                                    action_class=CompleteAction,
                                    _add_to_name=f"{type(self).__name__}")
        param = self.env.parameters
        param.ENV_DOES_REDISPATCHING = False
        self.env.change_parameters(param)
        self.obs = self.env.reset(seed=0, options={"time serie id": 0})
        return super().setUp()
    
    def tearDown(self):
        self.env.close()
        return super().tearDown()
    
    def test_issue(self):            
        obs, reward, done, info = self.env.step(self.env.action_space({"redispatch": [(0, -1)]}))
        assert obs.target_dispatch[0] == -1
        assert obs.actual_dispatch[0] == -1
    
    def test_storage(self):
        obs, reward, done, info = self.env.step(self.env.action_space({"set_storage": [(0, -1)]}))
        assert (np.abs(obs.target_dispatch) < 1e-6).all()
        assert (np.abs(obs.actual_dispatch) < 1e-6).all()
        assert obs.storage_power[0] == -1
        assert np.abs(obs.storage_charge[0] - (obs.storage_Emax[0] * 0.5 - 1/12 - obs.storage_loss[0] /12) ) < 1e-6
        assert np.abs(obs.storage_charge[1] - (obs.storage_Emax[1] * 0.5 - obs.storage_loss[1] /12) ) < 1e-6
        
    def test_curtailment(self):
        obs, reward, done, info = self.env.step(self.env.action_space({"curtail": [(2, 0.20)]}))
        assert (np.abs(obs.target_dispatch) < 1e-6).all()
        assert (np.abs(obs.actual_dispatch) < 1e-6).all()
        assert (obs.gen_p[2] - obs.gen_pmax[2] * 0.2) < 1e-6  # check curtailment is correctly applied
        
        obs, reward, done, info = self.env.step(self.env.action_space({"curtail": [(2, 1.)]}))
        assert (np.abs(obs.target_dispatch) < 1e-6).all()
        assert (np.abs(obs.actual_dispatch) < 1e-6).all()
        assert (obs.gen_p[2] - 17.) < 1e-6  # check curtailment is not put to 1. (target) but corresponds to the correct value
