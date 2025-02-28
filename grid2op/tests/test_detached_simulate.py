# Copyright (c) 2025, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import copy
import grid2op
from grid2op.Agent import BaseAgent
from grid2op.Action import CompleteAction
from grid2op.Chronics import ChangeNothing
import unittest
import warnings
import numpy as np
import pdb
# from pypowsybl2grid import PyPowSyBlBackend
# from lightsim2grid import LightSimBackend

class TestDetachSimulate(unittest.TestCase):
    def setUp(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make("educ_case14_storage",
                                    test=True,
                                    allow_detachment=True,
                                    action_class=CompleteAction,
                                    _add_to_name=type(self).__name__,
                                    chronics_class=ChangeNothing,
                                    data_feeding_kwargs={"h_forecast": [5, 10, 15, 20, 25, 30]},
                                    # backend=PyPowSyBlBackend()
                                    )
        paramerters = self.env.parameters
        paramerters.MAX_SUB_CHANGED = 99999
        paramerters.MAX_LINE_STATUS_CHANGED = 99999
        paramerters.NB_TIMESTEP_COOLDOWN_LINE = 0
        paramerters.NB_TIMESTEP_COOLDOWN_SUB = 0
        self.env.change_parameters(paramerters)
        self.init_obs = self.env.reset(seed=0, options={"time serie id": 0})
        
    def tearDown(self):
        self.env.close()
        return super().tearDown()
    
    def test_detach_gen(self, tol=1e-5):
        normal_v = 1. * self.init_obs.gen_v
        normal_p = 1. * self.init_obs.gen_p
        gen_id = 0
        act_deco = self.env.action_space({"set_bus": {"generators_id": [(gen_id, -1)]}})
        act_reco = self.env.action_space({"set_bus": {"generators_id": [(gen_id, +1)]}})
        next_obs, reward, done, info = self.env.step(act_deco)
        assert not done
        assert len(info["exception"]) == 0
        assert next_obs.gen_detached[gen_id]
        assert next_obs.gen_v[gen_id] == 0.
        assert next_obs.gen_p[gen_id] == 0.
        
        # test simulate
        next_obs_sim00, reward, done, info = next_obs.simulate(self.env.action_space())
        assert not done
        assert len(info["exception"]) == 0
        assert next_obs_sim00.gen_detached[gen_id]
        assert next_obs_sim00.gen_v[gen_id] == 0.
        assert next_obs_sim00.gen_p[gen_id] == 0.
        print("here here here")
        next_obs_sim0, reward, done, info = next_obs.simulate(act_reco)
        assert not done
        assert len(info["exception"]) == 0
        assert not next_obs_sim0.gen_detached[gen_id]
        assert np.abs(next_obs_sim0.gen_v[gen_id] - normal_v[gen_id]) <= tol, f"error {np.abs(next_obs_sim0.gen_v[gen_id] - normal_v[gen_id]).max()}"
        assert np.abs(next_obs_sim0.gen_p[gen_id] - normal_p[gen_id]) <= tol, f"error {np.abs(next_obs_sim0.gen_p[gen_id] - normal_p[gen_id]).max()}"
        next_obs_sim1, reward, done, info = next_obs.simulate(act_deco)
        assert not done
        assert len(info["exception"]) == 0
        assert next_obs_sim1.gen_detached[gen_id]
        assert next_obs_sim1.gen_v[gen_id] == 0.
        assert next_obs_sim1.gen_p[gen_id] == 0.
        next_obs_sim2, reward, done, info = next_obs.simulate(self.env.action_space())
        assert not done
        assert len(info["exception"]) == 0
        assert next_obs_sim2.gen_detached[gen_id]
        assert next_obs_sim2.gen_v[gen_id] == 0.
        assert next_obs_sim2.gen_p[gen_id] == 0.
        next_obs_sim3, reward, done, info = next_obs.simulate(act_reco)
        assert not done
        assert len(info["exception"]) == 0
        assert not next_obs_sim3.gen_detached[gen_id]
        assert np.abs(next_obs_sim3.gen_v[gen_id] - normal_v[gen_id]) <= tol, f"error {np.abs(next_obs_sim3.gen_v[gen_id] - normal_v[gen_id]).max()}"
        assert np.abs(next_obs_sim3.gen_p[gen_id] - normal_p[gen_id]) <= tol, f"error {np.abs(next_obs_sim3.gen_p[gen_id] - normal_p[gen_id]).max()}"
        # now change the setpoint
        next_obs_sim4, reward, done, info = next_obs.simulate(act_reco + self.env.action_space({"injection": {"prod_v": normal_v * 1.01}}))
        assert not done
        assert len(info["exception"]) == 0
        assert not next_obs_sim4.gen_detached[gen_id]
        assert np.abs(next_obs_sim4.gen_v[gen_id] - 1.01 * normal_v[gen_id]) <= tol, f"error {np.abs(next_obs_sim4.gen_v[gen_id] - 1.01 * normal_v[gen_id]).max()}"
        assert np.abs(next_obs_sim4.gen_p[gen_id] - normal_p[gen_id]) <= tol, f"error {np.abs(next_obs_sim4.gen_p[gen_id] - normal_p[gen_id]).max()}"
        # disco
        next_obs_sim5, reward, done, info = next_obs.simulate(act_deco)
        assert not done
        assert len(info["exception"]) == 0
        assert next_obs_sim5.gen_detached[gen_id]
        assert next_obs_sim5.gen_v[gen_id] == 0.
        assert next_obs_sim5.gen_p[gen_id] == 0.
        # reco again (and check the setpoint is not the last one used in obs.simulate but is the one of the observation)
        next_obs_sim6, reward, done, info = next_obs.simulate(act_reco)
        assert not done
        assert len(info["exception"]) == 0
        assert not next_obs_sim6.gen_detached[gen_id]
        assert np.abs(next_obs_sim6.gen_v[gen_id] - normal_v[gen_id]) <= tol, f"error {np.abs(next_obs_sim6.gen_v[gen_id] - normal_v[gen_id]).max()}"
        assert np.abs(next_obs_sim6.gen_p[gen_id] - normal_p[gen_id]) <= tol, f"error {np.abs(next_obs_sim6.gen_p[gen_id] - normal_p[gen_id]).max()}"
        
        
        # test step
        next_next_obs, reward, done, info = self.env.step(act_reco)
        assert not done
        assert len(info["exception"]) == 0
        assert not next_next_obs.gen_detached[gen_id]
        assert np.abs(next_next_obs.gen_v[gen_id] - normal_v[gen_id]) <= tol, f"error {np.abs(next_next_obs.gen_v[gen_id] - normal_v[gen_id]).max()}"
        assert np.abs(next_next_obs.gen_p[gen_id] - normal_p[gen_id]) <= tol, f"error {np.abs(next_next_obs.gen_p[gen_id] - normal_p[gen_id]).max()}"
        
    def test_detach_load(self, tol=1e-5):
        normal_q = 1. * self.init_obs.load_q
        normal_p = 1. * self.init_obs.load_p
        load_id = 0
        act_deco = self.env.action_space({"set_bus": {"loads_id": [(load_id, -1)]}})
        act_reco = self.env.action_space({"set_bus": {"loads_id": [(load_id, +1)]}})
        next_obs, reward, done, info = self.env.step(act_deco)
        assert not done
        assert len(info["exception"]) == 0
        assert next_obs.load_detached[load_id]
        assert next_obs.load_q[load_id] == 0.
        assert next_obs.load_p[load_id] == 0.
        
        # test simulate
        next_obs_sim00, reward, done, info = next_obs.simulate(self.env.action_space())
        assert not done
        assert len(info["exception"]) == 0
        assert next_obs_sim00.load_detached[load_id]
        assert next_obs_sim00.load_q[load_id] == 0.
        assert next_obs_sim00.load_p[load_id] == 0.
        
        next_obs_sim0, reward, done, info = next_obs.simulate(act_reco)
        assert not done
        assert len(info["exception"]) == 0
        assert not next_obs_sim0.load_detached[load_id]
        assert np.abs(next_obs_sim0.load_q[load_id] - normal_q[load_id]) <= tol, f"error {np.abs(next_obs_sim0.load_q[load_id] - normal_q[load_id]).max()}"
        assert np.abs(next_obs_sim0.load_p[load_id] - normal_p[load_id]) <= tol, f"error {np.abs(next_obs_sim0.load_p[load_id] - normal_p[load_id]).max()}"
        next_obs_sim1, reward, done, info = next_obs.simulate(act_deco)
        assert not done
        assert len(info["exception"]) == 0
        assert next_obs_sim1.load_detached[load_id]
        assert next_obs_sim1.load_q[load_id] == 0.
        assert next_obs_sim1.load_p[load_id] == 0.
        next_obs_sim2, reward, done, info = next_obs.simulate(self.env.action_space())
        assert not done
        assert len(info["exception"]) == 0
        assert next_obs_sim2.load_detached[load_id]
        assert next_obs_sim2.load_q[load_id] == 0.
        assert next_obs_sim2.load_p[load_id] == 0.
        next_obs_sim3, reward, done, info = next_obs.simulate(act_reco)
        assert not done
        assert len(info["exception"]) == 0
        assert not next_obs_sim3.load_detached[load_id]
        assert np.abs(next_obs_sim3.load_q[load_id] - normal_q[load_id]) <= tol, f"error {np.abs(next_obs_sim3.load_q[load_id] - normal_q[load_id]).max()}"
        assert np.abs(next_obs_sim3.load_p[load_id] - normal_p[load_id]) <= tol, f"error {np.abs(next_obs_sim3.load_p[load_id] - normal_p[load_id]).max()}"
        # now change the setpoint
        next_obs_sim4, reward, done, info = next_obs.simulate(act_reco + self.env.action_space({"injection": {"load_q": normal_q * 1.01}}))
        assert not done
        assert len(info["exception"]) == 0
        assert not next_obs_sim4.load_detached[load_id]
        assert np.abs(next_obs_sim4.load_q[load_id] - 1.01 * normal_q[load_id]) <= tol, f"error {np.abs(next_obs_sim4.load_q[load_id] - 1.01 * normal_q[load_id]).max()}"
        assert np.abs(next_obs_sim4.load_p[load_id] - normal_p[load_id]) <= tol, f"error {np.abs(next_obs_sim4.load_p[load_id] - normal_p[load_id]).max()}"
        # disco
        next_obs_sim5, reward, done, info = next_obs.simulate(act_deco)
        assert not done
        assert len(info["exception"]) == 0
        assert next_obs_sim5.load_detached[load_id]
        assert next_obs_sim5.load_q[load_id] == 0.
        assert next_obs_sim5.load_p[load_id] == 0.
        # reco again (and check the setpoint is not the last one used in obs.simulate but is the one of the observation)
        next_obs_sim6, reward, done, info = next_obs.simulate(act_reco)
        assert not done
        assert len(info["exception"]) == 0
        assert not next_obs_sim6.load_detached[load_id]
        assert np.abs(next_obs_sim6.load_q[load_id] - normal_q[load_id]) <= tol, f"error {np.abs(next_obs_sim6.load_q[load_id] - normal_q[load_id]).max()}"
        assert np.abs(next_obs_sim6.load_p[load_id] - normal_p[load_id]) <= tol, f"error {np.abs(next_obs_sim6.load_p[load_id] - normal_p[load_id]).max()}"
        
        # test step
        next_next_obs, reward, done, info = self.env.step(act_reco)
        assert not done
        assert len(info["exception"]) == 0
        assert not next_next_obs.load_detached[load_id]
        assert np.abs(next_next_obs.load_q[load_id] - normal_q[load_id]) <= tol, f"error {np.abs(next_next_obs.load_q[load_id] - normal_q[load_id]).max()}"
        assert np.abs(next_next_obs.load_p[load_id] - normal_p[load_id]) <= tol, f"error {np.abs(next_next_obs.load_p[load_id] - normal_p[load_id]).max()}"

        