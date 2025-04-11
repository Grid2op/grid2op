# Copyright (c) 2024, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.


import grid2op
import unittest
import warnings
import numpy as np
import pdb

import grid2op.Observation

class TestObsKirchhoff(unittest.TestCase):
    def setUp(self) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            # this needs to be tested with pandapower backend
            self.env = grid2op.make("l2rpn_idf_2023", test=True, _add_to_name=type(self).__name__)
        obs = self.env.reset(seed=0, options={"time serie id": 0})
    
    def tearDown(self) -> None:
        self.env.close()
        return super().tearDown()
    
    def _aux_normal_obs(self, obs: grid2op.Observation.BaseObservation, tol: float = 1e-4):
        p_subs, q_subs, p_bus, q_bus, diff_v_bus = obs.check_kirchhoff()
        assert np.abs(p_subs).max() <= tol, f"{np.abs(p_subs).max()}"
        assert np.abs(q_subs).max() <= tol, f"{np.abs(q_subs).max()}"
        assert np.abs(p_bus).max() <= tol, f"{np.abs(p_bus).max()}"
        assert np.abs(q_bus).max() <= tol, f"{np.abs(q_bus).max()}"
        assert np.abs(diff_v_bus).max() <= tol, f"{np.abs(diff_v_bus).max()}"
        
    def test_normal_obs(self):
        obs = self.env.reset(seed=0, options={"time serie id": 0})
        self._aux_normal_obs(obs)
        
        obs, *_ = self.env.step(self.env.action_space())
        self._aux_normal_obs(obs)

    def test_obs_set_line_status(self):
        obs = self.env.reset(seed=0, options={"time serie id": 0})
        line_id = 7
        obs, *_ = self.env.step(self.env.action_space({"set_line_status": [(line_id, -1)]}))
        self._aux_normal_obs(obs)
        
    def test_obs_set_bus(self):
        obs = self.env.reset(seed=0, options={"time serie id": 0})
        line_id = 7
        obs, *_ = self.env.step(self.env.action_space({"set_bus": {"lines_or_id": [(line_id, 2)]}}))
        self._aux_normal_obs(obs)
        
        
if __name__ == "__main__":
    unittest.main()
        