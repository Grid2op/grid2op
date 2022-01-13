# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import grid2op
import unittest
import warnings
from grid2op.Chronics import FromNPY
from grid2op.Exceptions import Grid2OpException
from grid2op.Runner import Runner
import numpy as np
import pdb


class TestNPYChronics(unittest.TestCase):
    """
    This class tests the possibility in grid2op to limit the number of call to "obs.simulate"
    """
    def setUp(self):
        self.env_name = "l2rpn_case14_sandbox"
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env_ref = grid2op.make(self.env_name)

        self.load_p = 1.0 * self.env_ref.chronics_handler.real_data.data.load_p
        self.load_q = 1.0 * self.env_ref.chronics_handler.real_data.data.load_q
        self.prod_p = 1.0 * self.env_ref.chronics_handler.real_data.data.prod_p
        self.prod_v = 1.0 * self.env_ref.chronics_handler.real_data.data.prod_v

    def test_proper_start_end(self):
        """test i can create an environment with the FromNPY class"""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = grid2op.make(self.env_name,
                               chronics_class=FromNPY,
                               data_feeding_kwargs={"i_start": 0,
                                                    "i_end": 18,  # excluded
                                                    "load_p": self.load_p,
                                                    "load_q": self.load_q,
                                                    "prod_p": self.prod_p,
                                                    "prod_v": self.prod_v}
                               )

        for ts in range(10):
            obs_ref, *_ = self.env_ref.step(env.action_space())
            assert np.all(obs_ref.gen_p[:-1] == self.prod_p[1 + ts, :-1]), f"error at iteration {ts}"
            obs, *_ = env.step(env.action_space())
            assert np.all(obs_ref.gen_p == obs.gen_p), f"error at iteration {ts}"

        # test the "end"
        for ts in range(7):
            obs, *_ = env.step(env.action_space())
        obs, reward, done, info = env.step(env.action_space())
        assert done
        with self.assertRaises(Grid2OpException):
            env.step(env.action_space())  # raises a Grid2OpException

    def test_proper_start_end_2(self):
        """test i can do as if the start was "later" """
        LAG = 5
        END = 18
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = grid2op.make(self.env_name,
                               chronics_class=FromNPY,
                               data_feeding_kwargs={"i_start": LAG,
                                                    "i_end": END,
                                                    "load_p": self.load_p,
                                                    "load_q": self.load_q,
                                                    "prod_p": self.prod_p,
                                                    "prod_v": self.prod_v}
                               )

        for ts in range(LAG):
            obs_ref, *_ = self.env_ref.step(env.action_space())

        for ts in range(END - LAG):
            obs_ref, *_ = self.env_ref.step(env.action_space())
            assert np.all(obs_ref.gen_p[:-1] == self.prod_p[1 + ts + LAG, :-1]), f"error at iteration {ts}"
            obs, *_ = env.step(env.action_space())
            assert np.all(obs_ref.gen_p == obs.gen_p), f"error at iteration {ts}"
        with self.assertRaises(Grid2OpException):
            env.step(env.action_space())  # raises a Grid2OpException because the env is done

    def test_iend_bigger_dim(self):
        max_step = 5
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = grid2op.make(self.env_name,
                               chronics_class=FromNPY,
                               data_feeding_kwargs={"i_start": 0,
                                                    "i_end": 10,  # excluded
                                                    "load_p": self.load_p[:max_step,:],
                                                    "load_q": self.load_q[:max_step,:],
                                                    "prod_p": self.prod_p[:max_step,:],
                                                    "prod_v": self.prod_v[:max_step,:]}
                               )
        assert env.chronics_handler.real_data.load_p.shape[0] == max_step
        for ts in range(max_step - 1):  # -1 because one ts is "burnt" for the initialization
            obs, reward, done, info = env.step(env.action_space())
            assert np.all(self.prod_p[1 + ts, :-1] == obs.gen_p[:-1]), f"error at iteration {ts}"
            
        obs, reward, done, info = env.step(env.action_space())
        assert done
        with self.assertRaises(Grid2OpException):
            env.step(env.action_space())  # raises a Grid2OpException because the env is done

    def test_change_chronics(self):
        """test i can change the chronics"""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = grid2op.make(self.env_name,
                               chronics_class=FromNPY,
                               data_feeding_kwargs={"i_start": 0,
                                                    "i_end": 18,  # excluded
                                                    "load_p": self.load_p,
                                                    "load_q": self.load_q,
                                                    "prod_p": self.prod_p,
                                                    "prod_v": self.prod_v}
                               )
        self.env_ref.reset()

        load_p = 1.0 * self.env_ref.chronics_handler.real_data.data.load_p
        load_q = 1.0 * self.env_ref.chronics_handler.real_data.data.load_q
        prod_p = 1.0 * self.env_ref.chronics_handler.real_data.data.prod_p
        prod_v = 1.0 * self.env_ref.chronics_handler.real_data.data.prod_v
        
        env.chronics_handler.real_data.change_chronics(load_p, load_q, prod_p, prod_v)
        for ts in range(10):
            obs, *_ = env.step(env.action_space())
            assert np.all(self.prod_p[1 + ts, :-1] == obs.gen_p[:-1]), f"error at iteration {ts}"
        env.reset()
        for ts in range(10):
            obs, *_ = env.step(env.action_space())
            assert np.all(prod_p[1 + ts, :-1] == obs.gen_p[:-1]), f"error at iteration {ts}"
    # TODO test runner
    # TODO test maintenance
    # TODO test hazards
    # TODO test forecasts

    # TODO test when env copied too !
    
if __name__ == "__main__":
    unittest.main()
