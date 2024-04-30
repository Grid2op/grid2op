# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.


import tempfile
import numpy as np
import warnings
import unittest

import grid2op
from grid2op.Episode import EpisodeData
from grid2op.Opponent import FromEpisodeDataOpponent
from grid2op.Runner import Runner
from grid2op.Action import TopologyAction, DispatchAction
from grid2op.tests.helper_path_test import *
from grid2op.Chronics import (FromHandlers,
                              Multifolder,
                              MultifolderWithCache,
                              GridStateFromFileWithForecasts,
                              GridStateFromFile,
                              GridStateFromFileWithForecastsWithMaintenance,
                              GridStateFromFileWithForecastsWithoutMaintenance,
                              FromOneEpisodeData,
                              FromMultiEpisodeData,
                              FromNPY)
from grid2op.Chronics.handlers import CSVHandler, JSONInitStateHandler

# TODO test forecast env
# TODO test with and without shunt
# TODO test grid2Op compat mode (storage units)
# TODO test with "names_orig_to_backend"
# TODO test with lightsimbackend
# TODO test with Runner
# TODO test other type of environment (multimix, masked etc.)


class TestSetActOrigDefault(unittest.TestCase):
    def _get_act_cls(self):
        return TopologyAction
    
    def _get_ch_cls(self):
        return Multifolder
    
    def _get_c_cls(self):
        return GridStateFromFileWithForecasts
    
    def _env_path(self):
        return os.path.join(
            PATH_DATA_TEST, "5bus_example_act_topo_set_init"
        )
    
    def setUp(self) -> None:
        self.env_nm = self._env_path()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make(self.env_nm,
                                    test=True,
                                    action_class=self._get_act_cls(),
                                    chronics_class=self._get_ch_cls(),
                                    data_feeding_kwargs={"gridvalueClass": self._get_c_cls()}
                                    )
        if issubclass(self._get_ch_cls(), MultifolderWithCache):
            self.env.chronics_handler.set_filter(lambda x: True)
            self.env.chronics_handler.reset()
        # some test to make sure the tests are correct
        assert issubclass(self.env.action_space.subtype, self._get_act_cls())
        assert isinstance(self.env.chronics_handler.real_data, self._get_ch_cls())
        assert isinstance(self.env.chronics_handler.real_data.data, self._get_c_cls())
        
    def tearDown(self) -> None:
        self.env.close()
        return super().tearDown()
    
    def test_working_setbus(self):
        # ts id 0 => set_bus
        self.obs = self.env.reset(seed=0, options={"time serie id": 0})
        
        assert self.obs.topo_vect[self.obs.line_or_pos_topo_vect[1]] == 2
        assert self.obs.topo_vect[self.obs.load_pos_topo_vect[0]] == 2
        assert (self.obs.time_before_cooldown_line == 0).all()
        assert (self.obs.time_before_cooldown_sub == 0).all()
        obs, reward, done, info = self.env.step(self.env.action_space())
        assert not done
        assert obs.topo_vect[self.obs.line_or_pos_topo_vect[1]] == 2
        assert obs.topo_vect[self.obs.load_pos_topo_vect[0]] == 2
        assert (obs.time_before_cooldown_line == 0).all()
        assert (obs.time_before_cooldown_sub == 0).all()
        

    def test_working_setstatus(self):
        # ts id 1 => set_status
        self.obs = self.env.reset(seed=0, options={"time serie id": 1})
        
        assert self.obs.topo_vect[self.obs.line_or_pos_topo_vect[1]] == -1
        assert self.obs.topo_vect[self.obs.line_ex_pos_topo_vect[1]] == -1
        assert not self.obs.line_status[1]
        assert (self.obs.time_before_cooldown_line == 0).all()
        assert (self.obs.time_before_cooldown_sub == 0).all()
        obs, reward, done, info = self.env.step(self.env.action_space())
        assert not done
        assert obs.topo_vect[self.obs.line_or_pos_topo_vect[1]] == -1
        assert obs.topo_vect[self.obs.line_ex_pos_topo_vect[1]] == -1
        assert not obs.line_status[1]
        assert (obs.time_before_cooldown_line == 0).all()
        assert (obs.time_before_cooldown_sub == 0).all()
        
    def test_rules_ok(self):
        """test that even if the action to set is illegal, it works (case of ts id 2)"""
        self.obs = self.env.reset(seed=0, options={"time serie id": 2})
        
        assert self.obs.topo_vect[self.obs.line_or_pos_topo_vect[1]] == 2
        assert self.obs.topo_vect[self.obs.line_ex_pos_topo_vect[5]] == 2
        assert (self.obs.time_before_cooldown_line == 0).all()
        assert (self.obs.time_before_cooldown_sub == 0).all()
        act_init = self.env.chronics_handler.get_init_action()
        obs, reward, done, info = self.env.step(act_init)
        assert info["exception"] is not None
        assert info["is_illegal"]
        assert obs.topo_vect[self.obs.line_or_pos_topo_vect[1]] == 2
        assert obs.topo_vect[self.obs.line_ex_pos_topo_vect[5]] == 2
        assert (obs.time_before_cooldown_line == 0).all()
        assert (obs.time_before_cooldown_sub == 0).all()


class TestSetActOrigDifferentActionCLS(TestSetActOrigDefault):
    def _get_act_cls(self):
        return DispatchAction


class TestSetAcOrigtMultiFolderWithCache(TestSetActOrigDefault):
    def _get_ch_cls(self):
        return MultifolderWithCache
    
    def test_two_reset_same(self):
        """test it does not crash when the same time series is used twice"""
        self.test_working_setstatus()
        obs, reward, done, info = self.env.step(self.env.action_space())
        self.test_working_setstatus()
        obs, reward, done, info = self.env.step(self.env.action_space())
    

class TestSetActOrigGridStateFromFile(TestSetActOrigDefault):
    def _get_c_cls(self):
        return GridStateFromFile
    
    
class TestSetActOrigGSFFWFWM(TestSetActOrigDefault):
    def _get_c_cls(self):
        return GridStateFromFileWithForecastsWithMaintenance
    
    
class TestSetActOrigGSFFWFWoM(TestSetActOrigDefault):
    def _get_c_cls(self):
        return GridStateFromFileWithForecastsWithoutMaintenance
    
    
class TestSetActOrigFromOneEpisodeData(TestSetActOrigDefault):
    def _aux_make_ep_data(self, ep_id):
        runner = Runner(**self.env.get_params_for_runner())
        runner.run(nb_episode=1,
                   episode_id=[ep_id],
                   path_save=self.fn.name,
                   max_iter=10)
        self.env.close()
        
        li_episode = EpisodeData.list_episode(self.fn.name)
        ep_data = li_episode[0]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make(self._env_path(),
                                    chronics_class=FromOneEpisodeData,
                                    data_feeding_kwargs={"ep_data": ep_data},
                                    opponent_class=FromEpisodeDataOpponent,
                                    opponent_attack_cooldown=1,
                                    )
        
    def setUp(self) -> None:
        self.fn = tempfile.TemporaryDirectory()
        super().setUp()
        
    def tearDown(self) -> None:
        self.fn.cleanup()
        return super().tearDown()
    
    def test_working_setbus(self):
        self._aux_make_ep_data(0)  # episode id 0 is used for this test
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            super().test_working_setbus()
        
    def test_working_setstatus(self):
        self._aux_make_ep_data(1)  # episode id 1 is used for this test
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            super().test_working_setstatus()
    
    def test_rules_ok(self):
        self._aux_make_ep_data(2)  # episode id 2 is used for this test
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            super().test_rules_ok()


class TestSetActOrigFromMultiEpisodeData(TestSetActOrigDefault):
    def setUp(self) -> None:
        super().setUp()
        self.fn = tempfile.TemporaryDirectory()
        runner = Runner(**self.env.get_params_for_runner())
        runner.run(nb_episode=3,
                   episode_id=[0, 1, 2],
                   path_save=self.fn.name,
                   max_iter=10)
        self.env.close()
        
        li_episode = EpisodeData.list_episode(self.fn.name)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make(self._env_path(),
                                    chronics_class=FromMultiEpisodeData,
                                    data_feeding_kwargs={"li_ep_data": li_episode},
                                    opponent_class=FromEpisodeDataOpponent,
                                    opponent_attack_cooldown=1,
                                    )
        
        
    def tearDown(self) -> None:
        self.fn.cleanup()
        return super().tearDown()
    
    def test_two_reset_same(self):
        """test it does not crash when the same time series is used twice"""
        self.test_working_setstatus()
        obs, reward, done, info = self.env.step(self.env.action_space())
        self.test_working_setstatus()
        obs, reward, done, info = self.env.step(self.env.action_space())
        
        
class TestSetActOrigFromNPY(TestSetActOrigDefault):
    def _aux_make_env(self, ch_id):
        self.obs = self.env.reset(seed=0, options={"time serie id": ch_id})
        load_p = 1.0 * self.env.chronics_handler._real_data.data.load_p[:self.max_iter,:]
        load_q = 1.0 * self.env.chronics_handler._real_data.data.load_q[:self.max_iter,:]
        gen_p = 1.0 * self.env.chronics_handler._real_data.data.prod_p[:self.max_iter,:]
        gen_v = np.repeat(self.obs.gen_v.reshape(1, -1), self.max_iter, axis=0)
        act = self.env.action_space({"set_bus": self.obs.topo_vect})
        self.env.close()
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make(self._env_path(),
                                    chronics_class=FromNPY,
                                    data_feeding_kwargs={"load_p": load_p,
                                                         "load_q": load_q,
                                                         "prod_p": gen_p,
                                                         "prod_v": gen_v,
                                                         "init_state": act
                                                         })
    def setUp(self) -> None:
        self.max_iter = 5
        super().setUp()
        
    def tearDown(self) -> None:
        return super().tearDown()
    
    def test_working_setbus(self):
        self._aux_make_env(0)  # episode id 0 is used for this test
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            super().test_working_setbus()
        
    def test_working_setstatus(self):
        self._aux_make_env(1)  # episode id 1 is used for this test
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            super().test_working_setstatus()
    
    def test_rules_ok(self):
        self._aux_make_env(2)  # episode id 2 is used for this test
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            super().test_rules_ok()
    

class TestSetActOrigEnvCopy(TestSetActOrigDefault):
    def setUp(self) -> None:
        super().setUp()
        env_cpy = self.env.copy()
        self.env.close()
        self.env = env_cpy


class TestSetActOrigFromHandlers(TestSetActOrigDefault):
    def setUp(self) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make(self._env_path(),
                                    data_feeding_kwargs={"gridvalueClass": FromHandlers,
                                                         "gen_p_handler": CSVHandler("prod_p"),
                                                         "load_p_handler": CSVHandler("load_p"),
                                                         "gen_v_handler": CSVHandler("prod_v"),
                                                         "load_q_handler": CSVHandler("load_q"),
                                                         "init_state_handler": JSONInitStateHandler("init_state_handler")
                                                        }
                                    )
    
    
if __name__ == "__main__":
    unittest.main()
