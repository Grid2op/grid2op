# Copyright (c) 2025, RTE (https://www.rte-france.com)
# See AUTHORS.txt and https://github.com/Grid2Op/grid2op/pull/319
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import numpy as np
import warnings
import grid2op
import unittest

from grid2op.Action import BaseAction
from grid2op.VoltageControler import ControlVoltageFromFile


class _TEST_VC_AgentOverride(ControlVoltageFromFile):
    def fix_voltage(self, observation, agent_action, env_action, prod_v_chronics):
        vect_ = None
        if "prod_v" in agent_action._dict_inj and np.isfinite(agent_action._dict_inj["prod_v"]).any():
            # agent decision
            vect_ = 1. * agent_action._dict_inj["prod_v"]
        if prod_v_chronics is not None:
            # default values (in the time series)
            if vect_ is None:
                # agent did not change anything
                vect_ = prod_v_chronics
            else:
                # keep the agent choice and put the default values
                # (from the time series) for the generator not modified
                # by the agent
                mask_default = ~np.isfinite(vect_)
                vect_[mask_default] = prod_v_chronics[mask_default]
                
        # now build the action
        if vect_ is not None:
            res = self.action_space({"injection": {"prod_v": vect_}})
        else:
            res = self.action_space()
            
        if observation is not None:
            # cache the get_topological_impact to avoid useless computations later
            # this is a speed optimization
            _ = res.get_topological_impact(observation.line_status, _store_in_cache=True, _read_from_cache=False)
            return res
        return super().fix_voltage(observation, agent_action, env_action, prod_v_chronics)


class TestIssue728(unittest.TestCase):
    def setUp(self) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make("educ_case14_storage",
                                    test=True,
                                    action_class=BaseAction,
                                    voltagecontroler_class=_TEST_VC_AgentOverride,
                                    _add_to_name=type(self).__name__)
        assert isinstance(self.env._voltage_controler, _TEST_VC_AgentOverride)
        assert issubclass(self.env._voltagecontrolerClass, _TEST_VC_AgentOverride)
        self.init_obs = self.env.reset(seed=0, options={"time serie id":0})
        return super().setUp()
    
    def test_can_modify_gen_v(self):
        new_prod_v = 1.0 * self.init_obs.gen_v
        new_prod_v[0] = 150.
        modify_prod_v_value = self.env.action_space({"injection": {"prod_v": new_prod_v}})
        obs, _, done, info = self.env.step(modify_prod_v_value)
        assert not done
        assert not info["is_ambiguous"]
        assert not info["is_illegal"]
        assert abs(obs.gen_v[0] - 150.) <= 1e-6
        
        
if __name__ == "__main__":
    unittest.main()
