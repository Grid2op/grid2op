# Copyright (c) 2024, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import datetime
import grid2op
import unittest
import warnings
import numpy as np
from grid2op.Exceptions import EnvError
from grid2op.gym_compat import GymEnv
from grid2op.Exceptions import Grid2OpException


class TestNewReset(unittest.TestCase):
    """
    This class tests the possibility to set the seed and the time
    serie id directly when calling `env.reset`
    """

    def setUp(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make("l2rpn_case14_sandbox", test=True, _add_to_name=type(self).__name__)
    
    def test_normal_env(self):
        # original way
        self.env.set_id(0)
        self.env.seed(0)
        obs = self.env.reset()
        
        # test with seed in reset
        self.env.set_id(0)
        obs_seed = self.env.reset(seed=0)
        
        # test with ts_id in reset
        self.env.seed(0)
        obs_ts = self.env.reset(options={"time serie id": 0})
        
        # test with both
        obs_both = self.env.reset(seed=0, options={"time serie id": 0})
        assert obs_seed == obs
        assert obs_ts == obs
        assert obs_both == obs
    
    def test_raise_if_wrong_key(self):
        with self.assertRaises(EnvError):
            obs_ts = self.env.reset(options={"time series id": 0})
            
        with self.assertRaises(EnvError):
            obs_ts = self.env.reset(options={"chronics id": 0})
    
    def _aux_obs_equals(self, obs1, obs2):
        assert obs1.keys() == obs2.keys(), f"not the same keys"
        for el in obs1:
            assert np.array_equal(obs1[el], obs2[el]), f"obs not equal for attribute {el}"
            
    def test_gym_env(self):
        gym_env = GymEnv(self.env)
        
        # original way (deprecated)
        # gym_env.init_env.set_id(0)
        # gym_env.init_env.seed(0)
        # obs, info = gym_env.reset()
        
        # test with seed in reset
        gym_env.init_env.set_id(0)
        obs_seed, info_seed = gym_env.reset(seed=0)
        
        # test with ts_id in reset
        gym_env.init_env.seed(0)
        obs_ts, info_ts = gym_env.reset(options={"time serie id": 0})
        
        # test with both
        obs_both, info_both = gym_env.reset(seed=0, options={"time serie id": 0})

        # self._aux_obs_equals(obs_seed, obs)
        self._aux_obs_equals(obs_ts, obs_seed)
        self._aux_obs_equals(obs_both, obs_seed)
        
    def test_init_datetime(self):
        # test from str
        ref_datetime = datetime.datetime(year=2024, month=12, day=6, hour=0, minute=0)
        obs_ts = self.env.reset(options={"init datetime": "2024-12-06 00:00"})
        assert obs_ts.year == 2024, f"{obs_ts.year} vs 2024"
        assert obs_ts.month == 12, f"{obs_ts.month} vs 12"
        assert obs_ts.day == 6, f"{obs_ts.day} vs 6"
        assert obs_ts.hour_of_day == 0, f"{obs_ts.hour_of_day} vs 0"
        assert obs_ts.minute_of_hour == 0, f"{obs_ts.minute_of_hour} vs 0"
        assert self.env.chronics_handler.start_datetime == ref_datetime
        assert self.env.chronics_handler.real_data.data.start_datetime == ref_datetime
        
        # test from datetime
        obs_ts = self.env.reset(options={"init datetime": ref_datetime})
        assert obs_ts.year == 2024, f"{obs_ts.year} vs 2024"
        assert obs_ts.month == 12, f"{obs_ts.month} vs 12"
        assert obs_ts.day == 6, f"{obs_ts.day} vs 6"
        assert obs_ts.hour_of_day == 0, f"{obs_ts.hour_of_day} vs 0"
        assert obs_ts.minute_of_hour == 0, f"{obs_ts.minute_of_hour} vs 0"
        assert self.env.chronics_handler.start_datetime == ref_datetime
        assert self.env.chronics_handler.real_data.data.start_datetime == ref_datetime
        
        # test an error is raised if format not understood
        with self.assertRaises(Grid2OpException):
            obs_ts = self.env.reset(options={"init datetime": 1})
        with self.assertRaises(Grid2OpException):
            obs_ts = self.env.reset(options={"init datetime": "06-12-2024 00:00"})
            
        # test when also skip ts is used
        obs_ts = self.env.reset(options={"init datetime": ref_datetime, "init ts": 12})
        assert obs_ts.year == 2024, f"{obs_ts.year} vs 2024"
        assert obs_ts.month == 12, f"{obs_ts.month} vs 12"
        assert obs_ts.day == 6, f"{obs_ts.day} vs 6"
        assert obs_ts.hour_of_day == 0, f"{obs_ts.hour_of_day} vs 0"
        assert obs_ts.minute_of_hour == 0, f"{obs_ts.minute_of_hour} vs 0"
        this_ref_next = ref_datetime - datetime.timedelta(hours=1) + self.env.chronics_handler.time_interval
        assert self.env.chronics_handler.real_data.data.start_datetime == this_ref_next
        
        # special case when skipping 1 step
        obs_ts = self.env.reset(options={"init datetime": ref_datetime, "init ts": 1})
        assert obs_ts.year == 2024, f"{obs_ts.year} vs 2024"
        assert obs_ts.month == 12, f"{obs_ts.month} vs 12"
        assert obs_ts.day == 6, f"{obs_ts.day} vs 6"
        assert obs_ts.hour_of_day == 0, f"{obs_ts.hour_of_day} vs 0"
        assert obs_ts.minute_of_hour == 0, f"{obs_ts.minute_of_hour} vs 0"
        assert self.env.chronics_handler.real_data.data.start_datetime == ref_datetime
        