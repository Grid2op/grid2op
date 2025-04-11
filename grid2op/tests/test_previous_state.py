# Copyright (c) 2025, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import copy
from typing import Union
import unittest
import warnings

import numpy as np
import grid2op
from grid2op.Action import BaseAction
import grid2op.Environment
import grid2op.Environment._env_prev_state
import grid2op.Observation


class PreviousStateTester(unittest.TestCase):
    def setUp(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make("educ_case14_storage",
                                    test=True,
                                    action_class=BaseAction,
                                    allow_detachment=True,
                                    _add_to_name=type(self).__name__)
        params = self.env.parameters
        params.NO_OVERFLOW_DISCONNECTION = True
        params.NB_TIMESTEP_COOLDOWN_LINE = 0
        params.NB_TIMESTEP_COOLDOWN_SUB = 0
        params.MAX_LINE_STATUS_CHANGED = 9999
        params.MAX_SUB_CHANGED = 9999
        self.env.change_parameters(params)
        self.env.change_forecast_parameters(params)
        self.obs = self.env.reset(seed=0, options={"time serie id": 0})  # for reproducibility
        return super().setUp()
    
    def tearDown(self):
        self.env.close()
        return super().tearDown()
    
    def _aux_test_matches_obs(self,
                              obs: Union[grid2op.Observation.BaseObservation, grid2op.Environment._env_prev_state._EnvPreviousState],
                              env: Union[grid2op.Environment.BaseEnv, grid2op.Environment._env_prev_state._EnvPreviousState],
                              extra: str = ""):
        obs_attrs = ["gen_p", "gen_v", "load_p", "load_q", "topo_vect", "storage_power"]
        prev_attrs = ["_gen_p", "_gen_v", "_load_p", "_load_q", "_topo_vect", "_storage_p"]
        if isinstance(obs, grid2op.Environment._env_prev_state._EnvPreviousState):
            this_attr = prev_attrs
        elif isinstance(obs, grid2op.Observation.BaseObservation):
            this_attr = obs_attrs
        else:
            raise RuntimeError("Test not implemented")
        
        if isinstance(env, grid2op.Environment.BaseEnv):
            env = env._previous_conn_state
            
        for ob_attr, pr_attr in zip(this_attr, prev_attrs):
            ob_ = getattr(obs, ob_attr)
            pr_ = getattr(env, pr_attr)
            assert np.allclose(ob_, pr_), f"{extra} error for attribute {ob_attr}: {ob_} vs {pr_}"

    def test_regular_step(self):
        self._aux_test_matches_obs(self.obs, self.env, "init state")
        obs, reward, done, info = self.env.step(self.env.action_space())
        self._aux_test_matches_obs(obs, self.env, "after 1st step")
        obs, reward, done, info = self.env.step(self.env.action_space())
        self._aux_test_matches_obs(obs, self.env, "after 2nd step")
    
    def test_storage(self):
        sto_id = 0
        pos_tv = self.env.storage_pos_topo_vect[0]
        obs, reward, done, info = self.env.step(self.env.action_space({"set_storage": [(sto_id, 1.)]}))
        assert obs.storage_power[sto_id] == 1.
        self._aux_test_matches_obs(obs, self.env, "after 1 step")
        
        # disconnect it
        obs, reward, done, info = self.env.step(self.env.action_space({"set_bus": {"storages_id": [(sto_id, -1)]}}))
        assert obs.storage_power[sto_id] == 0.
        assert obs.topo_vect[pos_tv] == -1
        assert self.env._previous_conn_state._topo_vect[pos_tv] == 1
        assert self.env._previous_conn_state._storage_p[sto_id] == 1.
        
        # reconnect it
        obs, reward, done, info = self.env.step(self.env.action_space({"set_bus": {"storages_id": [(sto_id, +1)]}}))
        assert obs.storage_power[sto_id] == 0.
        assert obs.topo_vect[pos_tv] == 1
        self._aux_test_matches_obs(obs, self.env, "after reco")
        
        # change_bus
        obs, reward, done, info = self.env.step(self.env.action_space({"set_bus": {"storages_id": [(sto_id, 2)],
                                                                                   "lines_or_id": [(7, 2)]}}))
        assert obs.topo_vect[pos_tv] == 2
        self._aux_test_matches_obs(obs, self.env, "after change_bus")
        
        # p setpoint
        obs, reward, done, info = self.env.step(self.env.action_space({"set_storage": [(sto_id, -1.)]}))
        assert obs.storage_power[sto_id] == -1.
        assert obs.topo_vect[pos_tv] == 2
        self._aux_test_matches_obs(obs, self.env, "after change bus and p setpoint")
        
        # disconnect it
        obs, reward, done, info = self.env.step(self.env.action_space({"set_bus": {"storages_id": [(sto_id, -1)]}}))
        assert obs.storage_power[sto_id] == 0.
        assert obs.topo_vect[pos_tv] == -1
        assert self.env._previous_conn_state._topo_vect[pos_tv] == 2
        assert self.env._previous_conn_state._storage_p[sto_id] == -1.
    
    def test_storage_simulate(self):
        # TODO
        pass
    
    def test_storage_for_env(self):
        # TODO
        pass
    
    def test_line_step(self):
        l_id = 0
        pos_or_tv = self.env.line_or_pos_topo_vect[0]
        pos_ex_tv = self.env.line_ex_pos_topo_vect[0]
        
        # just make sure the line l_id can be assign to buses 2,2 without creating another connected comp
        obs, reward, done, info = self.env.step(self.env.action_space({'set_bus': {"lines_or_id": [("1_3_3", 2)]}}))
        self._aux_test_matches_obs(obs, self.env, "after 1 step")
        
        obs, reward, done, info = self.env.step(self.env.action_space({'set_bus': {"lines_or_id": [(l_id, 2)]}}))
        assert obs.topo_vect[pos_or_tv] == 2
        assert obs.topo_vect[pos_ex_tv] == 1
        self._aux_test_matches_obs(obs, self.env, "after 2 step")
        
        obs, reward, done, info = self.env.step(self.env.action_space({'set_bus': {"lines_ex_id": [(l_id, 2)]}}))
        assert obs.topo_vect[pos_or_tv] == 2
        assert obs.topo_vect[pos_ex_tv] == 2
        self._aux_test_matches_obs(obs, self.env, "after 3 step")
        
        obs, reward, done, info = self.env.step(self.env.action_space({'set_bus': {"lines_or_id": [(l_id, 1)]}}))
        assert obs.topo_vect[pos_or_tv] == 1
        assert obs.topo_vect[pos_ex_tv] == 2
        self._aux_test_matches_obs(obs, self.env, "after 4 step")
        
        obs, reward, done, info = self.env.step(self.env.action_space({'set_bus': {"lines_ex_id": [(l_id, 1)]}}))
        assert obs.topo_vect[pos_or_tv] == 1
        assert obs.topo_vect[pos_ex_tv] == 1
        self._aux_test_matches_obs(obs, self.env, "after 5 step")
        
        # disconnect it
        obs, reward, done, info = self.env.step(self.env.action_space({'set_line_status': [(l_id, -1)]}))
        assert obs.topo_vect[pos_or_tv] == -1
        assert obs.topo_vect[pos_ex_tv] == -1
        assert self.env._previous_conn_state._topo_vect[pos_or_tv] == 1
        assert self.env._previous_conn_state._topo_vect[pos_ex_tv] == 1
        # reconnect it (step)
        obs, reward, done, info = self.env.step(self.env.action_space({'set_line_status': [(l_id, 1)]}))
        assert obs.topo_vect[pos_or_tv] == 1
        assert obs.topo_vect[pos_ex_tv] == 1
        assert self.env._previous_conn_state._topo_vect[pos_or_tv] == 1
        assert self.env._previous_conn_state._topo_vect[pos_ex_tv] == 1
        
        # do 2 steps above by changing bus of one side
        obs, reward, done, info = self.env.step(self.env.action_space({'set_bus': {"lines_or_id": [(l_id, 2)]}}))
        obs, reward, done, info = self.env.step(self.env.action_space({'set_line_status': [(l_id, -1)]}))
        assert obs.topo_vect[pos_or_tv] == -1
        assert obs.topo_vect[pos_ex_tv] == -1
        assert self.env._previous_conn_state._topo_vect[pos_or_tv] == 2
        assert self.env._previous_conn_state._topo_vect[pos_ex_tv] == 1
        # reconnect it (step)
        obs, reward, done, info = self.env.step(self.env.action_space({'set_line_status': [(l_id, 1)]}))
        assert obs.topo_vect[pos_or_tv] == 2
        assert obs.topo_vect[pos_ex_tv] == 1
        assert self.env._previous_conn_state._topo_vect[pos_or_tv] == 2
        assert self.env._previous_conn_state._topo_vect[pos_ex_tv] == 1
        
        # do 2 steps above by changing bus of the other side side
        obs, reward, done, info = self.env.step(self.env.action_space({'set_bus': {"lines_ex_id": [(l_id, 2)]}}))
        obs, reward, done, info = self.env.step(self.env.action_space({'set_line_status': [(l_id, -1)]}))
        assert obs.topo_vect[pos_or_tv] == -1
        assert obs.topo_vect[pos_ex_tv] == -1
        assert self.env._previous_conn_state._topo_vect[pos_or_tv] == 2
        assert self.env._previous_conn_state._topo_vect[pos_ex_tv] == 2
        # reconnect it (step)
        obs, reward, done, info = self.env.step(self.env.action_space({'set_line_status': [(l_id, 1)]}))
        assert obs.topo_vect[pos_or_tv] == 2
        assert obs.topo_vect[pos_ex_tv] == 2
        assert self.env._previous_conn_state._topo_vect[pos_or_tv] == 2
        assert self.env._previous_conn_state._topo_vect[pos_ex_tv] == 2
        
    def test_line_simulate(self):
        l_id = 0
        pos_or_tv = self.env.line_or_pos_topo_vect[0]
        pos_ex_tv = self.env.line_ex_pos_topo_vect[0]
        
        # just make sure the line l_id can be assign to buses 2,2 without creating another connected comp
        obs, reward, done, info = self.env.step(self.env.action_space({'set_bus': {"lines_or_id": [("1_3_3", 2)]}}))
        self._aux_test_matches_obs(obs, self.env, "after 1 step")
        
        # disconnect it
        obs, reward, done, info = self.env.step(self.env.action_space({'set_line_status': [(l_id, -1)]}))
        assert obs.topo_vect[pos_or_tv] == -1
        assert obs.topo_vect[pos_ex_tv] == -1
        assert self.env._previous_conn_state._topo_vect[pos_or_tv] == 1
        assert self.env._previous_conn_state._topo_vect[pos_ex_tv] == 1
        # reconnect it (simulate)
        sim_obs, *_  = obs.simulate(self.env.action_space({'set_line_status': [(l_id, 1)]}))
        assert sim_obs.topo_vect[pos_or_tv] == 1
        assert sim_obs.topo_vect[pos_ex_tv] == 1
        # reconnect it (step it should match the simulate above)
        obs, reward, done, info = self.env.step(self.env.action_space({'set_line_status': [(l_id, 1)]}))
        assert obs.topo_vect[pos_or_tv] == 1
        assert obs.topo_vect[pos_ex_tv] == 1
        
        # do 2 steps above by changing bus of one side
        obs, reward, done, info = self.env.step(self.env.action_space({'set_bus': {"lines_or_id": [(l_id, 2)]}}))
        obs, reward, done, info = self.env.step(self.env.action_space({'set_line_status': [(l_id, -1)]}))
        assert obs.topo_vect[pos_or_tv] == -1
        assert obs.topo_vect[pos_ex_tv] == -1
        assert self.env._previous_conn_state._topo_vect[pos_or_tv] == 2
        assert self.env._previous_conn_state._topo_vect[pos_ex_tv] == 1
        # reconnect it (simulate)
        sim_obs, *_  = obs.simulate(self.env.action_space({'set_line_status': [(l_id, 1)]}))
        assert sim_obs.topo_vect[pos_or_tv] == 2
        assert sim_obs.topo_vect[pos_ex_tv] == 1
        sim_obs2, *_ = obs.simulate(self.env.action_space({'set_bus': {"lines_or_id": [(l_id, 1)]}}))
        assert sim_obs2.topo_vect[pos_or_tv] == 1
        assert sim_obs2.topo_vect[pos_ex_tv] == 1
        sim_obs3, *_  = obs.simulate(self.env.action_space({'set_line_status': [(l_id, 1)]}))
        assert sim_obs3.topo_vect[pos_or_tv] == 2 # should not be affected by the previous simulate
        assert sim_obs3.topo_vect[pos_ex_tv] == 1
        # reconnect it (step - it should match the last simulate above)
        obs, reward, done, info = self.env.step(self.env.action_space({'set_line_status': [(l_id, 1)]}))
        assert obs.topo_vect[pos_or_tv] == 2
        assert obs.topo_vect[pos_ex_tv] == 1
        
        # do 2 steps above by changing bus of the other side side
        obs, reward, done, info = self.env.step(self.env.action_space({'set_bus': {"lines_ex_id": [(l_id, 2)]}}))
        obs, reward, done, info = self.env.step(self.env.action_space({'set_line_status': [(l_id, -1)]}))
        assert obs.topo_vect[pos_or_tv] == -1
        assert obs.topo_vect[pos_ex_tv] == -1
        assert self.env._previous_conn_state._topo_vect[pos_or_tv] == 2
        assert self.env._previous_conn_state._topo_vect[pos_ex_tv] == 2
        # reconnect it (simulate)
        sim_obs, *_  = obs.simulate(self.env.action_space({'set_line_status': [(l_id, 1)]}))
        assert sim_obs.topo_vect[pos_or_tv] == 2
        assert sim_obs.topo_vect[pos_ex_tv] == 2
        sim_obs2, *_ = obs.simulate(self.env.action_space({'set_bus': {"lines_ex_id": [(l_id, 1)]}}))
        assert sim_obs2.topo_vect[pos_or_tv] == 2
        assert sim_obs2.topo_vect[pos_ex_tv] == 1
        sim_obs3, *_  = obs.simulate(self.env.action_space({'set_line_status': [(l_id, 1)]}))
        assert sim_obs3.topo_vect[pos_or_tv] == 2 
        assert sim_obs3.topo_vect[pos_ex_tv] == 2 # should not be affected by the previous simulate
        # reconnect it (step  - it should match the last simulate above)
        obs, reward, done, info = self.env.step(self.env.action_space({'set_line_status': [(l_id, 1)]}))
        assert obs.topo_vect[pos_or_tv] == 2
        assert obs.topo_vect[pos_ex_tv] == 2
        assert self.env._previous_conn_state._topo_vect[pos_or_tv] == 2
        assert self.env._previous_conn_state._topo_vect[pos_ex_tv] == 2
        
    def test_line_for_env(self):
        l_id = 0
        pos_or_tv = self.env.line_or_pos_topo_vect[0]
        pos_ex_tv = self.env.line_ex_pos_topo_vect[0]
        
        # just make sure the line l_id can be assign to buses 2,2 without creating another connected comp
        obs, reward, done, info = self.env.step(self.env.action_space({'set_bus': {"lines_or_id": [("1_3_3", 2)]}}))
        self._aux_test_matches_obs(obs, self.env, "after 1 step")
        
        # disconnect it
        obs, reward, done, info = self.env.step(self.env.action_space({'set_line_status': [(l_id, -1)]}))
        assert obs.topo_vect[pos_or_tv] == -1
        assert obs.topo_vect[pos_ex_tv] == -1
        assert self.env._previous_conn_state._topo_vect[pos_or_tv] == 1
        assert self.env._previous_conn_state._topo_vect[pos_ex_tv] == 1
        # reconnect it (forecast env)
        for_env = obs.get_forecast_env()
        for_obs = for_env.reset()
        assert for_obs.topo_vect[pos_or_tv] == -1
        assert for_obs.topo_vect[pos_ex_tv] == -1
        for_obs2, *_ = for_env.step(self.env.action_space({'set_line_status': [(l_id, 1)]}))
        assert for_obs2.topo_vect[pos_or_tv] == 1
        assert for_obs2.topo_vect[pos_ex_tv] == 1
        
        # reconnect it (step) - should match forecast env
        obs, reward, done, info = self.env.step(self.env.action_space({'set_line_status': [(l_id, 1)]}))
        assert obs.topo_vect[pos_or_tv] == 1
        assert obs.topo_vect[pos_ex_tv] == 1
        assert self.env._previous_conn_state._topo_vect[pos_or_tv] == 1
        assert self.env._previous_conn_state._topo_vect[pos_ex_tv] == 1
        
        # do 2 steps above by changing bus of one side
        obs, reward, done, info = self.env.step(self.env.action_space({'set_bus': {"lines_or_id": [(l_id, 2)]}}))
        obs, reward, done, info = self.env.step(self.env.action_space({'set_line_status': [(l_id, -1)]}))
        assert obs.topo_vect[pos_or_tv] == -1
        assert obs.topo_vect[pos_ex_tv] == -1
        assert self.env._previous_conn_state._topo_vect[pos_or_tv] == 2
        assert self.env._previous_conn_state._topo_vect[pos_ex_tv] == 1
        # reconnect it (forecast env)
        for_env = obs.get_forecast_env()
        assert for_env._cst_prev_state_at_init._topo_vect[pos_or_tv] == 2, f"{for_env._cst_prev_state_at_init._topo_vect[pos_or_tv]} vs 2"
        assert for_env._previous_conn_state._topo_vect[pos_or_tv] == 2, f"{for_env._previous_conn_state._topo_vect[pos_or_tv]} vs 2"
        assert for_env._backend_action.last_topo_registered.values[pos_or_tv] == 2, f"{for_env._backend_action.last_topo_registered.values[pos_or_tv]} vs 2"
        
        for_obs = for_env.reset()
        assert for_env._cst_prev_state_at_init._topo_vect[pos_or_tv] == 2, f"{for_env._cst_prev_state_at_init._topo_vect[pos_or_tv]} vs 2"
        assert for_env._previous_conn_state._topo_vect[pos_or_tv] == 2, f"{for_env._previous_conn_state._topo_vect[pos_or_tv]} vs 2"
        assert for_env._backend_action.last_topo_registered.values[pos_or_tv] == 2, f"{for_env._backend_action.last_topo_registered.values[pos_or_tv]} vs 2"
        assert for_obs.topo_vect[pos_or_tv] == -1
        assert for_obs.topo_vect[pos_ex_tv] == -1
        for_obs2, *_ = for_env.step(self.env.action_space({'set_line_status': [(l_id, 1)]}))
        assert for_obs2.topo_vect[pos_or_tv] == 2
        assert for_obs2.topo_vect[pos_ex_tv] == 1
        # reconnect it (step) - should match forecast env
        obs, reward, done, info = self.env.step(self.env.action_space({'set_line_status': [(l_id, 1)]}))
        assert obs.topo_vect[pos_or_tv] == 2
        assert obs.topo_vect[pos_ex_tv] == 1
        assert self.env._previous_conn_state._topo_vect[pos_or_tv] == 2
        assert self.env._previous_conn_state._topo_vect[pos_ex_tv] == 1
        
        # do 2 steps above by changing bus of the other side side
        obs, reward, done, info = self.env.step(self.env.action_space({'set_bus': {"lines_ex_id": [(l_id, 2)]}}))
        obs, reward, done, info = self.env.step(self.env.action_space({'set_line_status': [(l_id, -1)]}))
        assert obs.topo_vect[pos_or_tv] == -1
        assert obs.topo_vect[pos_ex_tv] == -1
        assert self.env._previous_conn_state._topo_vect[pos_or_tv] == 2
        assert self.env._previous_conn_state._topo_vect[pos_ex_tv] == 2
        # reconnect it (forecast env)
        for_env = obs.get_forecast_env()
        assert for_env._backend_action.last_topo_registered.values[pos_or_tv] == 2
        assert for_env._backend_action.last_topo_registered.values[pos_ex_tv] == 2
        for_obs = for_env.reset()
        assert for_env._backend_action.last_topo_registered.values[pos_or_tv] == 2
        assert for_env._backend_action.last_topo_registered.values[pos_ex_tv] == 2
        assert for_obs.topo_vect[pos_or_tv] == -1
        assert for_obs.topo_vect[pos_ex_tv] == -1
        for_obs2, *_ = for_env.step(self.env.action_space({'set_line_status': [(l_id, 1)]}))
        assert for_obs2.topo_vect[pos_or_tv] == 2
        assert for_obs2.topo_vect[pos_ex_tv] == 2
        # reconnect it (step) - should match forecast env
        obs, reward, done, info = self.env.step(self.env.action_space({'set_line_status': [(l_id, 1)]}))
        assert obs.topo_vect[pos_or_tv] == 2
        assert obs.topo_vect[pos_ex_tv] == 2
        assert self.env._previous_conn_state._topo_vect[pos_or_tv] == 2
        assert self.env._previous_conn_state._topo_vect[pos_ex_tv] == 2
    
    def test_cst_prev_state_const(self, env=None):
        if env is None:
            env = self.env
            
        assert not env._cst_prev_state_at_init._can_modif
        # at init, assert there is no nan basically
        self._aux_test_matches_obs(env._cst_prev_state_at_init, env._cst_prev_state_at_init, "after 0 step")
        
        init_cst_prev_state = env._cst_prev_state_at_init
        init_cst_prev_state_cpy = copy.deepcopy(env._cst_prev_state_at_init)
        _ = env.step(env.action_space())
        assert env._cst_prev_state_at_init is init_cst_prev_state
        assert not env._cst_prev_state_at_init._can_modif
        self._aux_test_matches_obs(init_cst_prev_state_cpy, env._cst_prev_state_at_init, "after 1 step")
        _ = env.step(env.action_space({'set_line_status': [(0, -1)]}))
        assert env._cst_prev_state_at_init is init_cst_prev_state
        assert not env._cst_prev_state_at_init._can_modif
        self._aux_test_matches_obs(init_cst_prev_state_cpy, env._cst_prev_state_at_init, "after 2 steps")
        
        _ = env.reset()
        assert env._cst_prev_state_at_init is init_cst_prev_state
        assert not env._cst_prev_state_at_init._can_modif
        self._aux_test_matches_obs(init_cst_prev_state_cpy, env._cst_prev_state_at_init, "after 1 reset")
        _ = env.step(env.action_space())
        assert env._cst_prev_state_at_init is init_cst_prev_state
        assert not env._cst_prev_state_at_init._can_modif
        self._aux_test_matches_obs(init_cst_prev_state_cpy, env._cst_prev_state_at_init, "after 1 reset 1 step")
        _ = env.step(env.action_space({'set_line_status': [(0, -1)]}))
        assert env._cst_prev_state_at_init is init_cst_prev_state
        assert not env._cst_prev_state_at_init._can_modif
        self._aux_test_matches_obs(init_cst_prev_state_cpy, env._cst_prev_state_at_init, "after 1 reset 2 steps")
        
        _ = env.reset()
        assert env._cst_prev_state_at_init is init_cst_prev_state
        assert not env._cst_prev_state_at_init._can_modif
        self._aux_test_matches_obs(init_cst_prev_state_cpy, env._cst_prev_state_at_init, "after 2 reset")
        _ = env.step(env.action_space())
        assert env._cst_prev_state_at_init is init_cst_prev_state
        assert not env._cst_prev_state_at_init._can_modif
        self._aux_test_matches_obs(init_cst_prev_state_cpy, env._cst_prev_state_at_init, "after 2 reset 1 step")
        _ = env.step(env.action_space({'set_line_status': [(0, -1)]}))
        assert env._cst_prev_state_at_init is init_cst_prev_state
        assert not env._cst_prev_state_at_init._can_modif
        self._aux_test_matches_obs(init_cst_prev_state_cpy, env._cst_prev_state_at_init, "after 2 reset 2 steps")
        
    def test_cst_prev_state_const_for_env(self, env=None):
        obs = self.env.reset(seed=0, options={"time serie id": 0})
        self.test_cst_prev_state_const(env=obs.get_forecast_env())
        
    def test_shunt(self, tol=1e-5):
        sh_id = 0
        th_val_p = [0., 0., 0., 0., 0., 0.]
        th_val_q = [-21.010567, -21.036667, -21.025122, -21.041656, -21.038073, -21.049952]

        obs, reward, done, info = self.env.step(self.env.action_space())
        assert obs._shunt_bus[sh_id] == 1
        assert np.abs(obs._shunt_p[sh_id] - th_val_p[obs.current_step]) <= tol, f"{obs._shunt_p[sh_id]} vs {th_val_p[obs.current_step]}"
        assert np.abs(obs._shunt_q[sh_id] - th_val_q[obs.current_step]) <= tol, f"{obs._shunt_p[sh_id]} vs {th_val_q[obs.current_step]}"
        self._aux_test_matches_obs(obs, self.env, "after 1 step")
        
        # for battery of tests: the bus
        obs, reward, done, info = self.env.step(self.env.action_space({"shunt": {"set_bus": [(sh_id, -1)]}}))
        id_pr_ok = obs.current_step - 1
        assert obs._shunt_bus[sh_id] == -1
        assert obs._shunt_p[sh_id] == 0.
        assert obs._shunt_q[sh_id] == 0.
        assert self.env._previous_conn_state._shunt_bus[sh_id] == 1
        assert np.abs(self.env._previous_conn_state._shunt_p - th_val_p[id_pr_ok]) <= tol, f"{obs._shunt_p[sh_id]} vs {th_val_p[id_pr_ok]}"
        assert np.abs(self.env._previous_conn_state._shunt_q - th_val_q[id_pr_ok]) <= tol, f"{obs._shunt_p[sh_id]} vs {th_val_q[id_pr_ok]}"
        
        obs, reward, done, info = self.env.step(self.env.action_space({"shunt": {"set_bus": [(sh_id, 1)]}}))
        assert obs._shunt_bus[sh_id] == 1
        assert np.abs(obs._shunt_p[sh_id] - th_val_p[obs.current_step]) <= tol, f"{obs._shunt_p[sh_id]} vs {th_val_p[obs.current_step]}"
        assert np.abs(obs._shunt_q[sh_id] - th_val_q[obs.current_step]) <= tol, f"{obs._shunt_p[sh_id]} vs {th_val_q[obs.current_step]}"
        assert self.env._previous_conn_state._shunt_bus[sh_id] == 1
        
    def test_shunt_simulate(self):
        # TODO
        pass
    
    def test_shunt_for_env(self):
        # TODO
        pass
    
        
# TODO shunt with obs.simulate and env.get_forecast_env
# TODO storage with obs.simulate and env.get_forecast_env
        
        
# LATER: 
# TODO test when load / gen / storage is disco then reco
   # - when no new setpoint
   # - when setpoint has previously changed
   # - when new setpoint is set
# TODO test_line with chained forecast -- not sure of the desired behaviour to be honest...