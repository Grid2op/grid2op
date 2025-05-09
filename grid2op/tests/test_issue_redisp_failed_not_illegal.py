# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import unittest
import warnings
import grid2op

class IssueRedispTester(unittest.TestCase):
    def setUp(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env  = grid2op.make("educ_case14_storage", test=True, _add_to_name=type(self).__name__)
        params = env.parameters
        params.NO_OVERFLOW_DISCONNECTION = True
        env.change_parameters(params)
        env_cls = type(env)
        for el in [env, env_cls]:
            el.gen_pmax[-1] = 280.
            el.gen_max_ramp_down[-1] = 280.
            el.gen_max_ramp_up[-1] = 280.
        self.env = env
        return super().setUp()

    def test_redisp_too_often(self):
        obs = self.env.reset(options={"time serie id": 0})
        act = self.env.action_space({"redispatch": [(0, -5)]})
        for nb_step in range(28):
            obs, reward, done, info = self.env.step(act)
            assert not done
            assert not info["failed_redispatching"]
            
        obs, reward, done, info = self.env.step(act)
        assert not done
        assert info["failed_redispatching"]
        
        act = self.env.action_space({"redispatch": [(0, 5)]})
        obs, reward, done, info = self.env.step(act)
        assert not done
        assert not info["failed_redispatching"]

if __name__ == "__main__":
    unittest.main()
