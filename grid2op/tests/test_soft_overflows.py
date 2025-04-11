# Copyright (c) 2025, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import unittest
import warnings

import numpy as np
import grid2op


class TestSoftOverflow(unittest.TestCase):
    def setUp(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make("l2rpn_case14_sandbox", test=True)
        self.init_obs = self.env.reset(seed=0, options={"time serie id": 0})
        obs, *_ = self.env.step(self.env.action_space())
        self.th_lim = 2. * obs.a_or
        mask_ = self.init_obs.a_or > obs.a_or
        self.th_lim[mask_] = 2. * self.init_obs.a_or[mask_]
        self.line_id = 1
        assert not mask_[self.line_id ]  # flow on line self.line_id  should increase
        self.th_lim[self.line_id ] = self.init_obs.a_or[self.line_id ] + 1e-5
        # flow on line self.line_id  should be overflown when t >= 1
        return super().setUp()
    
    def tearDown(self):
        self.env.close()
        return super().tearDown()
    
    def test_whenO(self):
        params = self.env.parameters
        params.NB_TIMESTEP_OVERFLOW_ALLOWED = 0
        self.env.change_parameters(params)
        obs0 = self.env.reset(seed=0, options={"time serie id": 0, "thermal limit": self.th_lim})
        assert obs0.a_or[self.line_id ] > 1e-5
        assert obs0.line_status[self.line_id]
        assert obs0.timestep_overflow[self.line_id] == 0
        
        obs, *_ = self.env.step(self.env.action_space())
        assert np.abs(obs.a_or[self.line_id]) <= 1e-8
        assert not obs.line_status[self.line_id]
        assert obs.timestep_overflow[self.line_id] == 0
        
    def test_when1(self):
        params = self.env.parameters
        params.NB_TIMESTEP_OVERFLOW_ALLOWED = 1
        self.env.change_parameters(params)
        obs0 = self.env.reset(seed=0, options={"time serie id": 0, "thermal limit": self.th_lim})
        assert obs0.a_or[self.line_id] > 1e-5
        assert obs0.line_status[self.line_id]
        assert obs0.timestep_overflow[self.line_id] == 0
        
        obs1, *_ = self.env.step(self.env.action_space())
        assert np.abs(obs1.a_or[self.line_id]) > 1e-8
        assert obs1.line_status[self.line_id]
        assert obs1.timestep_overflow[self.line_id] == 1
        
        obs2, *_ = self.env.step(self.env.action_space())
        assert np.abs(obs2.a_or[self.line_id]) <= 1e-8
        assert not obs2.line_status[self.line_id]
        assert obs2.timestep_overflow[self.line_id] == 0
        
    def test_when2(self):
        params = self.env.parameters
        params.NB_TIMESTEP_OVERFLOW_ALLOWED = 2
        self.env.change_parameters(params)
        obs0 = self.env.reset(seed=0, options={"time serie id": 0, "thermal limit": self.th_lim})
        assert obs0.a_or[self.line_id] > 1e-5
        assert obs0.line_status[self.line_id]
        assert obs0.timestep_overflow[self.line_id] == 0
        
        obs1, *_ = self.env.step(self.env.action_space())
        assert np.abs(obs1.a_or[self.line_id]) > 1e-8
        assert obs1.line_status[self.line_id]
        assert obs1.timestep_overflow[self.line_id] == 1
        
        obs2, *_ = self.env.step(self.env.action_space())
        assert np.abs(obs2.a_or[self.line_id]) > 1e-8
        assert obs2.line_status[self.line_id]
        assert obs2.timestep_overflow[self.line_id] == 2
        
        obs3, *_ = self.env.step(self.env.action_space())
        assert np.abs(obs3.a_or[self.line_id]) <= 1e-8
        assert not obs3.line_status[self.line_id]
        assert obs3.timestep_overflow[self.line_id] == 0
        
    def test_when3(self):
        params = self.env.parameters
        params.NB_TIMESTEP_OVERFLOW_ALLOWED = 3
        self.env.change_parameters(params)
        obs0 = self.env.reset(seed=0, options={"time serie id": 0, "thermal limit": self.th_lim})
        assert obs0.a_or[self.line_id] > 1e-5
        assert obs0.line_status[self.line_id]
        assert obs0.timestep_overflow[self.line_id] == 0
        
        obs1, *_ = self.env.step(self.env.action_space())
        assert np.abs(obs1.a_or[self.line_id]) > 1e-8
        assert obs1.line_status[self.line_id]
        assert obs1.timestep_overflow[self.line_id] == 1
        
        obs2, *_ = self.env.step(self.env.action_space())
        assert np.abs(obs2.a_or[self.line_id]) > 1e-8
        assert obs2.line_status[self.line_id]
        assert obs2.timestep_overflow[self.line_id] == 2
        
        obs3, *_ = self.env.step(self.env.action_space())
        assert np.abs(obs3.a_or[self.line_id]) > 1e-8
        assert obs3.line_status[self.line_id]
        assert obs3.timestep_overflow[self.line_id] == 3
        
        obs4, *_ = self.env.step(self.env.action_space())
        assert np.abs(obs4.a_or[self.line_id]) <= 1e-8
        assert not obs4.line_status[self.line_id]
        assert obs4.timestep_overflow[self.line_id] == 0
        
    def test_when4(self):
        params = self.env.parameters
        params.NB_TIMESTEP_OVERFLOW_ALLOWED = 4
        self.env.change_parameters(params)
        obs0 = self.env.reset(seed=0, options={"time serie id": 0, "thermal limit": self.th_lim})
        assert obs0.a_or[self.line_id] > 1e-5
        assert obs0.line_status[self.line_id]
        assert obs0.timestep_overflow[self.line_id] == 0
        
        obs1, *_ = self.env.step(self.env.action_space())
        assert np.abs(obs1.a_or[self.line_id]) > 1e-8
        assert obs1.line_status[self.line_id]
        assert obs1.timestep_overflow[self.line_id] == 1
        
        obs2, *_ = self.env.step(self.env.action_space())
        assert np.abs(obs2.a_or[self.line_id]) > 1e-8
        assert obs2.line_status[self.line_id]
        assert obs2.timestep_overflow[self.line_id] == 2
        
        obs3, *_ = self.env.step(self.env.action_space())
        assert np.abs(obs3.a_or[self.line_id]) > 1e-8
        assert obs3.line_status[self.line_id]
        assert obs3.timestep_overflow[self.line_id] == 3
        
        obs4, *_ = self.env.step(self.env.action_space())
        assert np.abs(obs4.a_or[self.line_id]) > 1e-8
        assert obs4.line_status[self.line_id]
        assert obs4.timestep_overflow[self.line_id] == 4
        
        obs5, *_ = self.env.step(self.env.action_space())
        assert np.abs(obs5.a_or[self.line_id]) <= 1e-8
        assert not obs5.line_status[self.line_id]
        assert obs5.timestep_overflow[self.line_id] == 0
        
    def test_when5(self):
        params = self.env.parameters
        params.NB_TIMESTEP_OVERFLOW_ALLOWED = 5
        self.env.change_parameters(params)
        obs0 = self.env.reset(seed=0, options={"time serie id": 0, "thermal limit": self.th_lim})
        assert obs0.a_or[self.line_id] > 1e-5
        assert obs0.line_status[self.line_id]
        assert obs0.timestep_overflow[self.line_id] == 0
        
        obs1, *_ = self.env.step(self.env.action_space())
        assert np.abs(obs1.a_or[self.line_id]) > 1e-8
        assert obs1.line_status[self.line_id]
        assert obs1.timestep_overflow[self.line_id] == 1
        
        obs2, *_ = self.env.step(self.env.action_space())
        assert np.abs(obs2.a_or[self.line_id]) > 1e-8
        assert obs2.line_status[self.line_id]
        assert obs2.timestep_overflow[self.line_id] == 2
        
        obs3, *_ = self.env.step(self.env.action_space())
        assert np.abs(obs3.a_or[self.line_id]) > 1e-8
        assert obs3.line_status[self.line_id]
        assert obs3.timestep_overflow[self.line_id] == 3
        
        obs4, *_ = self.env.step(self.env.action_space())
        assert np.abs(obs4.a_or[self.line_id]) > 1e-8
        assert obs4.line_status[self.line_id]
        assert obs4.timestep_overflow[self.line_id] == 4
        
        obs5, *_ = self.env.step(self.env.action_space())
        assert np.abs(obs5.a_or[self.line_id]) > 1e-8
        assert obs5.line_status[self.line_id]
        assert obs5.timestep_overflow[self.line_id] == 5
        
        obs6, *_ = self.env.step(self.env.action_space())
        assert np.abs(obs6.a_or[self.line_id]) <= 1e-8
        assert not obs6.line_status[self.line_id]
        assert obs6.timestep_overflow[self.line_id] == 0
        