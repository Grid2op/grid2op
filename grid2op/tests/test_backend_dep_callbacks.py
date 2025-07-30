# Copyright (c) 2025, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

# do some generic tests that can be implemented directly to test if a backend implementation can work out of the box
# with grid2op.
# see an example of test_Pandapower for how to use this suit.
import copy
import unittest
import numpy as np
import warnings

import grid2op
from grid2op.Parameters import Parameters
from grid2op.Action import BaseAction
from grid2op.Exceptions import InvalidBackendCallback
import pdb


class Test_BackendDepCallbacks(unittest.TestCase):
    def setUp(self) -> None:
        self.env_name = "educ_case14_storage"
        param = Parameters()
        param.NO_OVERFLOW_DISCONNECTION = True
        param.NB_TIMESTEP_COOLDOWN_LINE = 0
        param.NB_TIMESTEP_COOLDOWN_SUB = 0
        param.ACTIVATE_STORAGE_LOSS = False
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make(
                self.env_name, test=True, action_class=BaseAction, param=param,
                _add_to_name=type(self).__name__
            )

    def tearDown(self) -> None:
        self.env.close()
        
    def test_act_dn_call_back_dn(self):
        _ = self.env.reset(seed=0, options={"time serie id": 0})
        env_cpy = self.env.copy()
        act = self.env.action_space()
        act.backend_dependant_callback = lambda pp_grid: None
        obs, reward, *_ = self.env.step(act)
        obs_cpy, reward_cpy, *_ = env_cpy.step(self.env.action_space())
        assert abs(reward_cpy - reward) <= 1e-5
        assert np.abs(obs.load_p - obs_cpy.load_p).max() <= 1e-5
        assert np.abs(obs.gen_p - obs_cpy.gen_p).max() <= 1e-5
        assert np.abs(obs.a_or - obs_cpy.a_or).max() <= 1e-5
        assert np.abs(obs.v_or - obs_cpy.v_or).max() <= 1e-5
        
    def test_callback_persistent(self):
        _ = self.env.reset(seed=0, options={"time serie id": 0})
        env_cpy = self.env.copy()
        act = self.env.action_space()
        def tmp_callback(pp_grid):
            pp_grid.trafo.loc[0, "tap_pos"] = 1
        act.backend_dependant_callback = tmp_callback
        
        # check callback is performed
        obs, reward, *_ = self.env.step(act)
        obs_cpy, reward_cpy, *_ = env_cpy.step(self.env.action_space())
        assert abs(reward_cpy - reward) > 0.5
        assert np.abs(obs.v_or - obs_cpy.v_or).max() > 0.6
        assert self.env.backend._grid.trafo.loc[0, "tap_pos"] == 1
        assert env_cpy.backend._grid.trafo.loc[0, "tap_pos"] == -1
    
        # check callback is still present even after a "do nothing"
        obs, reward, *_ = self.env.step(self.env.action_space())
        obs_cpy, reward_cpy, *_ = env_cpy.step(self.env.action_space())
        assert abs(reward_cpy - reward) > 0.5
        assert np.abs(obs.v_or - obs_cpy.v_or).max() > 0.6
        assert self.env.backend._grid.trafo.loc[0, "tap_pos"] == 1
        assert env_cpy.backend._grid.trafo.loc[0, "tap_pos"] == -1
        
    def test_callback_properly_reset(self):    
        _ = self.env.reset(seed=0, options={"time serie id": 0})
        env_cpy = self.env.copy()
        act = self.env.action_space()
        def tmp_callback(pp_grid):
            pp_grid.trafo.loc[0, "tap_pos"] = 1
        act.backend_dependant_callback = tmp_callback
        
        # check callback is performed
        obs, reward, *_ = self.env.step(act)
        obs_cpy, reward_cpy, *_ = env_cpy.step(self.env.action_space())
        assert abs(reward_cpy - reward) > 0.5
        assert np.abs(obs.v_or - obs_cpy.v_or).max() > 0.6
        assert self.env.backend._grid.trafo.loc[0, "tap_pos"] == 1
        assert env_cpy.backend._grid.trafo.loc[0, "tap_pos"] == -1
            
        # check callbacks is properly removed
        obs = self.env.reset(seed=1, options={"time serie id": 0})
        obs_cpy = env_cpy.reset(seed=1, options={"time serie id": 0})
        assert np.abs(obs.load_p - obs_cpy.load_p).max() <= 1e-5
        assert np.abs(obs.gen_p - obs_cpy.gen_p).max() <= 1e-5
        assert np.abs(obs.a_or - obs_cpy.a_or).max() <= 1e-5
        assert np.abs(obs.v_or - obs_cpy.v_or).max() <= 1e-5
        assert self.env.backend._grid.trafo.loc[0, "tap_pos"] == -1
        assert env_cpy.backend._grid.trafo.loc[0, "tap_pos"] == -1
        
    def test_invalid_callback_do_nothing(self):
        _ = self.env.reset(seed=0, options={"time serie id": 0})
        env_cpy = self.env.copy()
        env_cpy_2 = self.env.copy()
        act = self.env.action_space()
        act.line_or_set_bus = [(0, 2)]  # force the action is not "do nothing"
        act_cpy_2 = act.copy()
        
        def tmp_callback(pp_grid):
            raise RuntimeError("Invalid callback on purpose")
        act.backend_dependant_callback = tmp_callback
        
        obs, reward, done, info = self.env.step(act)
        obs_cpy, reward_cpy, done_cpy, info_cpy = env_cpy.step(self.env.action_space())
        obs_cpy_2, reward_cpy_2, done_cpy_2, info_cpy_2 = env_cpy_2.step(act_cpy_2)
        assert not done
        assert not done_cpy
        assert not done_cpy_2
        
        # info properly updated
        assert len(info["exception"])
        assert isinstance(info["exception"][1], InvalidBackendCallback)
        assert info["is_ambiguous"]
        assert info_cpy["exception"] == []
        assert info_cpy_2["exception"] == []
        # bus is properly modified only in the second case
        assert obs.line_or_bus[0] == 1
        assert obs_cpy.line_or_bus[0] == 1
        assert obs_cpy_2.line_or_bus[0] == 2
        
        # obs and obs_cpy are the same
        assert np.abs(obs.load_p - obs_cpy.load_p).max() <= 1e-5
        assert np.abs(obs.gen_p - obs_cpy.gen_p).max() <= 1e-5
        assert np.abs(obs.a_or - obs_cpy.a_or).max() <= 1e-5
        assert np.abs(obs.v_or - obs_cpy.v_or).max() <= 1e-5
        
        # obs (do nothing) and obs_cpy_2 (set_bus effective) are not the same
        assert np.abs(obs.a_or - obs_cpy_2.a_or).max() > 0.5
        
    def test_action_copy(self):
        # create an action with a callback
        act = self.env.action_space()
        act.line_or_set_bus = [(0, 2)]  # force the action is not "do nothing"
        
        def tmp_callback(pp_grid):
            raise RuntimeError("Invalid callback on purpose")
        act.backend_dependant_callback = tmp_callback
        
        # grid2op copy
        act_cpy = act.copy()
        assert act_cpy.backend_dependant_callback is act.backend_dependant_callback
        
        # copy.copy
        act_cpy_2 = copy.copy(act)
        assert act_cpy_2.backend_dependant_callback is act.backend_dependant_callback
        
        # copy.deepcopy
        act_cpy_3 = copy.deepcopy(act)
        assert act_cpy_3.backend_dependant_callback is act.backend_dependant_callback
