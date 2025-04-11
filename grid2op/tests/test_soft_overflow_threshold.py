# Copyright (c) 2023, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import grid2op
from grid2op.Parameters import Parameters
import warnings
import unittest


class TestSoftOverflowThreshold(unittest.TestCase):    
    def setUp(self) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make("l2rpn_case14_sandbox", test=True, _add_to_name=type(self).__name__)
        self.env.seed(0)
        self.env.set_id(0)
        th_lim = self.env.get_thermal_limit()
        th_lim[0] = 161
        self.env.set_thermal_limit(th_lim)
    
    def tearDown(self) -> None:
        self.env.close()
        return super().tearDown()
    
    def test_default_param(self):
        """test nothing is broken, and by default it works normally"""
        obs = self.env.reset()
        obs, *_ = self.env.step(self.env.action_space())
        obs, *_ = self.env.step(self.env.action_space())
        obs, *_ = self.env.step(self.env.action_space())
        assert not obs.line_status[0] 
    
    def test_1point1_param_nodisc(self):
        """test line is NOT disconnected when its flow is bellow the threshold"""
        param = self.env.parameters
        param.SOFT_OVERFLOW_THRESHOLD = 1.1
        self.env.change_parameters(param)
        obs = self.env.reset()
        obs, *_ = self.env.step(self.env.action_space())
        obs, *_ = self.env.step(self.env.action_space())
        obs, *_ = self.env.step(self.env.action_space())
        assert obs.line_status[0] 
        assert obs.timestep_overflow[0] == 3 
        assert obs.thermal_limit[0] == 161
        assert obs.a_or[0] > 161
    
    def test_1point1_param_disco(self):
        """test line is indeed disconnected when its flow is above the threshold"""
        param = self.env.parameters
        param.SOFT_OVERFLOW_THRESHOLD = 1.1
        self.env.change_parameters(param)
        th_lim = self.env.get_thermal_limit()
        th_lim[0] /= 1.1
        self.env.set_thermal_limit(th_lim)
        obs = self.env.reset()
        obs, *_ = self.env.step(self.env.action_space())
        obs, *_ = self.env.step(self.env.action_space())
        obs, *_ = self.env.step(self.env.action_space())
        assert not obs.line_status[0] 


class TestSoftOverflowThresholdBigger1(unittest.TestCase):    
    def setUp(self) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make("l2rpn_case14_sandbox", test=True, _add_to_name=type(self).__name__)
        self.env.seed(0)
        self.env.set_id(0)
        th_lim = self.env.get_thermal_limit()
        th_lim[0] = 161
        self.env.set_thermal_limit(th_lim)
    
    def tearDown(self) -> None:
        self.env.close()
        return super().tearDown()
    
    def test_ts_overflow_not_equal_prot_eng(self):
        """test that the timestep_overflow is properly updated (1 per step)
        but that the timestep_protection_engaged is not (in this case because SOFT_OVERFLOW_THRESHOLD
        higher than limit)
        """
        param = self.env.parameters
        param.SOFT_OVERFLOW_THRESHOLD = 1.1
        self.env.change_parameters(param)
        th_lim = self.env.get_thermal_limit()
        th_lim[0] /= 1.05
        self.env.set_thermal_limit(th_lim)
        obs = self.env.reset()
        obs, *_ = self.env.step(self.env.action_space())
        assert obs.timestep_overflow[0] == 1
        assert obs.timestep_protection_engaged[0] == 0
        obs, *_ = self.env.step(self.env.action_space())
        assert obs.timestep_overflow[0] == 2
        assert obs.timestep_protection_engaged[0] == 0
        obs, *_ = self.env.step(self.env.action_space())
        assert obs.timestep_overflow[0] == 3
        assert obs.timestep_protection_engaged[0] == 0
        obs, *_ = self.env.step(self.env.action_space())
        assert obs.timestep_overflow[0] == 4
        assert obs.timestep_protection_engaged[0] == 0
        
    def test_protection_correctly_engaged(self):
        """
        test that the time overcurrent protection is properly engaged at the right step
        """
        param = self.env.parameters
        param.SOFT_OVERFLOW_THRESHOLD = 1.1
        param.NB_TIMESTEP_OVERFLOW_ALLOWED = 2
        self.env.change_parameters(param)
        th_lim = self.env.get_thermal_limit()
        th_lim[1] *= 0.295 / 1.1
        self.env.set_thermal_limit(th_lim)
        obs = self.env.reset(options={"time serie id": 0})
        
        obs, *_ = self.env.step(self.env.action_space())
        assert abs(obs.rho[1] - 1.10686) <= 1e-5
        assert obs.timestep_overflow[0] == 1
        assert obs.timestep_overflow[1] == 1
        assert obs.timestep_protection_engaged[1] == 1
        
        obs, *_ = self.env.step(self.env.action_space())
        assert abs(obs.rho[1] - 1.09062) <= 1e-5
        assert obs.timestep_overflow[0] == 2
        assert obs.timestep_overflow[1] == 2
        assert obs.timestep_protection_engaged[1] == 0
        
        obs, *_ = self.env.step(self.env.action_space())
        assert abs(obs.rho[1] - 1.0933142) <= 1e-5
        assert obs.timestep_overflow[0] == 3
        assert obs.timestep_overflow[1] == 3
        assert obs.timestep_protection_engaged[1] == 0
        
        obs, *_ = self.env.step(self.env.action_space())
        assert abs(obs.rho[1] - 1.11505) <= 1e-5
        assert obs.timestep_overflow[0] == 4
        assert obs.timestep_overflow[1] == 4
        assert obs.timestep_protection_engaged[1] == 1
        
        obs, *_ = self.env.step(self.env.action_space())
        assert abs(obs.rho[1] - 1.12392) <= 1e-5
        assert obs.timestep_overflow[0] == 5
        assert obs.timestep_overflow[1] == 5
        assert obs.timestep_protection_engaged[1] == 2
        
        obs, *_ = self.env.step(self.env.action_space())
        assert abs(obs.rho[1] - 0.) <= 1e-5
        assert not obs.line_status[1]
        assert obs.timestep_overflow[0] == 6
        assert obs.timestep_overflow[1] == 0
        assert obs.timestep_protection_engaged[1] == 0
        
        
class TestSoftOverflowThresholdLower1(unittest.TestCase):    
    def setUp(self) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make("l2rpn_case14_sandbox", test=True, _add_to_name=type(self).__name__)
        self.env.seed(0)
        self.env.set_id(0)
        th_lim = self.env.get_thermal_limit()
        th_lim[0] = 161
        self.env.set_thermal_limit(th_lim)
    
    def tearDown(self) -> None:
        self.env.close()
        return super().tearDown()
    
    def test_ts_overflow_not_equal_prot_eng(self):
        """test that the timestep_protection_engaged is properly updated (1 per step)
        but that the timestep_protectimestep_overflowtion_engaged is not 
        (in this case because SOFT_OVERFLOW_THRESHOLD lower than 1)
        """
        param = self.env.parameters
        param.SOFT_OVERFLOW_THRESHOLD = 0.9
        param.NO_OVERFLOW_DISCONNECTION = True
        self.env.change_parameters(param)
        th_lim = self.env.get_thermal_limit()
        th_lim[0] *= 1.1
        self.env.set_thermal_limit(th_lim)
        obs = self.env.reset()
        obs, *_ = self.env.step(self.env.action_space())
        assert obs.timestep_overflow[0] == 0
        assert obs.timestep_protection_engaged[0] == 1
        obs, *_ = self.env.step(self.env.action_space())
        assert obs.timestep_overflow[0] == 0
        assert obs.timestep_protection_engaged[0] == 2
        obs, *_ = self.env.step(self.env.action_space())
        assert obs.timestep_overflow[0] == 0
        assert obs.timestep_protection_engaged[0] == 3
        obs, *_ = self.env.step(self.env.action_space())
        assert obs.timestep_overflow[0] == 0
        assert obs.timestep_protection_engaged[0] == 4
        
    def test_ts_overflow_not_equal_prot_eng_disco_of(self):
        """test that the timestep_protection_engaged is properly updated (1 per step)
        but that the timestep_protectimestep_overflowtion_engaged is not 
        (in this case because SOFT_OVERFLOW_THRESHOLD lower than 1)
        """
        param = self.env.parameters
        param.SOFT_OVERFLOW_THRESHOLD = 0.9
        param.NO_OVERFLOW_DISCONNECTION = False
        self.env.change_parameters(param)
        th_lim = self.env.get_thermal_limit()
        th_lim[0] *= 1.1
        self.env.set_thermal_limit(th_lim)
        obs = self.env.reset()
        obs, *_ = self.env.step(self.env.action_space())
        assert obs.timestep_overflow[0] == 0
        assert obs.timestep_protection_engaged[0] == 1
        obs, *_ = self.env.step(self.env.action_space())
        assert obs.timestep_overflow[0] == 0
        assert obs.timestep_protection_engaged[0] == 2
        obs, *_ = self.env.step(self.env.action_space())
        assert abs(obs.rho[0] - 0.) <= 1e-5
        assert not obs.line_status[0]
        
    def test_protection_correctly_engaged(self):
        """
        test that the time overcurrent protection is properly engaged at the right step
        """
        param = self.env.parameters
        param.SOFT_OVERFLOW_THRESHOLD = 0.915
        param.NB_TIMESTEP_OVERFLOW_ALLOWED = 2
        self.env.change_parameters(param)
        th_lim = self.env.get_thermal_limit()
        th_lim[0] *= 1.1
        self.env.set_thermal_limit(th_lim)
        obs = self.env.reset(options={"time serie id": 0})
        
        obs, *_ = self.env.step(self.env.action_space())
        assert abs(obs.rho[0] - 0.951674) <= 1e-5
        assert obs.timestep_overflow[0] == 0
        assert obs.timestep_protection_engaged[0] == 1
        
        obs, *_ = self.env.step(self.env.action_space())
        assert abs(obs.rho[0] - 0.9156) <= 1e-5
        assert obs.timestep_overflow[0] == 0
        assert obs.timestep_protection_engaged[0] == 2
        
        obs, *_ = self.env.step(self.env.action_space())
        assert abs(obs.rho[0] - 0.914302) <= 1e-5
        assert obs.timestep_overflow[0] == 0
        assert obs.timestep_protection_engaged[0] == 0
        
        obs, *_ = self.env.step(self.env.action_space())
        assert abs(obs.rho[0] - 0.950629) <= 1e-5
        assert obs.timestep_overflow[0] == 0
        assert obs.timestep_protection_engaged[0] == 1
        
        obs, *_ = self.env.step(self.env.action_space())
        assert abs(obs.rho[0] - 0.956885) <= 1e-5
        assert obs.timestep_overflow[0] == 0
        assert obs.timestep_protection_engaged[0] == 2
        
        obs, *_ = self.env.step(self.env.action_space())
        assert abs(obs.rho[0] - 0.) <= 1e-5
        assert not obs.line_status[0]
        assert obs.timestep_overflow[0] == 0
        assert obs.timestep_protection_engaged[0] == 0
    
    
if __name__ == '__main__':
    unittest.main()
