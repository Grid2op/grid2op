# Copyright (c) 2024, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.


from datetime import timedelta
import numpy as np
import grid2op
import unittest
import warnings
from grid2op.Chronics import ChangeNothing

import pdb

class TestForecastEnvTester(unittest.TestCase):
    def setUp(self) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            # this needs to be tested with pandapower backend
            self.env = grid2op.make("l2rpn_idf_2023", test=True, _add_to_name=type(self).__name__)
        obs = self.env.reset(seed=0, options={"time serie id": 0})
    
    def tearDown(self) -> None:
        self.env.close()
        return super().tearDown()
    
    def _aux_normal_obs(self, obs, line_id=0):
        for_env = obs.get_forecast_env()
        for_obs = for_env.reset()
        assert (for_obs.topo_vect == obs.topo_vect).all(), f"{(for_obs.topo_vect != obs.topo_vect).nonzero()}"
        
        for_obs = for_env.reset(options={"init state": {"set_line_status": [(line_id, -1)]}})
        assert (for_obs.topo_vect != obs.topo_vect).sum() == 2
        assert for_obs.topo_vect[type(self.env).line_or_pos_topo_vect[line_id]] == -1
        assert for_obs.topo_vect[type(self.env).line_ex_pos_topo_vect[line_id]] == -1
        
        for_obs = for_env.reset(options={"init state": {"set_bus": {"lines_or_id": [(line_id, 2)]}}})
        assert (for_obs.topo_vect != obs.topo_vect).sum() == 1
        assert for_obs.topo_vect[type(self.env).line_or_pos_topo_vect[line_id]] == 2
        
    def test_normal_obs(self):
        obs = self.env.reset(seed=0, options={"time serie id": 0})
        self._aux_normal_obs(obs)
        
        obs, *_ = self.env.step(self.env.action_space())
        self._aux_normal_obs(obs)

    def test_obs_set_line_status(self):
        obs = self.env.reset(seed=0, options={"time serie id": 0})
        line_id = 7
        obs, *_ = self.env.step(self.env.action_space({"set_line_status": [(line_id, -1)]}))
        self._aux_normal_obs(obs, line_id=0)
        
    def test_obs_set_bus(self):
        obs = self.env.reset(seed=0, options={"time serie id": 0})
        line_id = 7
        obs, *_ = self.env.step(self.env.action_space({"set_bus": {"lines_or_id": [(line_id, 2)]}}))
        self._aux_normal_obs(obs, line_id=0)
        
        
class TestDoNothingAndForecastEnv(unittest.TestCase):
    def setUp(self):
        dict_ = {"chronics_class": ChangeNothing,
                 "data_feeding_kwargs": {
                                         "h_forecast": [i * 30 for i in range(1, 48)],  # Every 30 mins ahead up for 47 timesteps
                                         "time_interval": timedelta(minutes=30),
                                         }}
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make("l2rpn_case14_sandbox",
                                    test=True,
                                    **dict_
                                    )
        return super().setUp()
    
    def tearDown(self):
        self.env.close()
        return super().tearDown()
    
    def test_forecast_env(self, tol=1e-5):
        obs = self.env.reset(seed=0, options={"time serie id": 0})
        # get the forecast env
        params = self.env.parameters
        params.NO_OVERFLOW_DISCONNECTION = True
        for_env = obs.get_forecast_env()
        for_env.change_parameters(params)
        # check that forecast observations matches env data
        sim_obs = for_env.reset()
        assert (np.abs(obs.gen_p - sim_obs.gen_p) <= tol).all(), f"error for {i}: {np.abs(obs.gen_p - sim_obs.gen_p).max()}"
        assert (np.abs(obs.load_p - sim_obs.load_p) <= tol).all(), f"error for {i}: {np.abs(obs.load_p - sim_obs.load_p).max()}"
        assert (np.abs(obs.load_q - sim_obs.load_q) <= tol).all(), f"error for {i}: {np.abs(obs.load_q - sim_obs.load_q).max()}"
        assert (np.abs(obs.a_or - sim_obs.a_or) <= 100. * tol).all(), f"error for {i}: {np.abs(obs.a_or - sim_obs.a_or).max()}"
            
        for i in range(47):
            sim_obs, sim_r, sim_d, sim_i = for_env.step(self.env.action_space())
            assert not sim_d, f"error for {i}"
            assert (np.abs(obs.gen_p - sim_obs.gen_p) <= tol).all(), f"error for {i}: {np.abs(obs.gen_p - sim_obs.gen_p).max()}"
            assert (np.abs(obs.load_p - sim_obs.load_p) <= tol).all(), f"error for {i}: {np.abs(obs.load_p - sim_obs.load_p).max()}"
            assert (np.abs(obs.load_q - sim_obs.load_q) <= tol).all(), f"error for {i}: {np.abs(obs.load_q - sim_obs.load_q).max()}"
            assert (np.abs(obs.a_or - sim_obs.a_or) <= 100. * tol).all(), f"error for {i}: {np.abs(obs.a_or - sim_obs.a_or).max()}"
    
    
if __name__ == "__main__":
    unittest.main()
        