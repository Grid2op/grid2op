# Copyright (c) 2025, RTE (https://www.rte-france.com)
# See AUTHORS.txt
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
from grid2op.VoltageControler import VCFromFileAgentOverrides


class TestVCFromFileAO(unittest.TestCase):
    def setUp(self) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make("educ_case14_storage",
                                    test=True,
                                    action_class=BaseAction,
                                    voltagecontroler_class=VCFromFileAgentOverrides,
                                    _add_to_name=type(self).__name__)
        assert isinstance(self.env._voltage_controler, VCFromFileAgentOverrides)
        assert issubclass(self.env._voltagecontrolerClass, VCFromFileAgentOverrides)
        self.init_obs = self.env.reset(seed=0, options={"time serie id":0})
        return super().setUp()
    
    def test_can_modify_one_gen_v(self):
        new_prod_v = 1.0 * self.init_obs.gen_v
        new_prod_v[0] = 150.
        modify_prod_v_value = self.env.action_space({"injection": {"prod_v": new_prod_v}})
        obs, _, done, info = self.env.step(modify_prod_v_value)
        assert not done
        assert not info["is_ambiguous"]
        assert not info["is_illegal"]
        assert abs(obs.gen_v[0] - 150.) <= 1e-6
        assert abs(obs.gen_v[1:] - self.init_obs.gen_v[1:]).max() <= 1e-6
    
    def test_can_modify_all_gen_v(self):
        new_prod_v = 1.05 * self.init_obs.gen_v
        modify_prod_v_value = self.env.action_space({"injection": {"prod_v": new_prod_v}})
        obs, _, done, info = self.env.step(modify_prod_v_value)
        assert not done
        assert not info["is_ambiguous"]
        assert not info["is_illegal"]
        assert abs(obs.gen_v - 1.05 * self.init_obs.gen_v).max() <= 1e-6
