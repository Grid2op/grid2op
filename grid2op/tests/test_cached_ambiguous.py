# Copyright (c) 2025, RTE (https://www.rte-france.com)
# See AUTHORS.txt and https://github.com/Grid2Op/grid2op/pull/319
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import warnings

import numpy as np
import grid2op
from grid2op.Action import BaseAction
import unittest


class TestCachedAmbiguous(unittest.TestCase):
    def setUp(self) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make("educ_case14_storage",
                                    test=True,
                                    action_class=BaseAction,
                                    allow_detachment=True,
                                    _add_to_name=type(self).__name__)
        self.init_obs = self.env.reset(seed=0, options={"time serie id":0})
        return super().setUp()
    
    def tearDown(self):
        self.env.close()
        return super().tearDown()
    
    def test_cached_default(self):
        act = self.env.action_space()
        assert act._cached_is_not_ambiguous
    
    def test_cache_set(self):
        act = self.env.action_space()
        act._cached_is_not_ambiguous = False
        act.is_ambiguous()
        assert act._cached_is_not_ambiguous
        
    def test_modif_bool(self):
        act = self.env.action_space()
        act.change_bus = 0
        assert not act._cached_is_not_ambiguous
        
        act = self.env.action_space()
        act.change_bus = np.array([0])
        assert not act._cached_is_not_ambiguous
        
        act = self.env.action_space()
        act.change_bus = np.zeros(self.env.dim_topo, dtype=bool)
        assert not act._cached_is_not_ambiguous
        
        act = self.env.action_space()
        act.change_bus = [0]
        assert not act._cached_is_not_ambiguous
        
        act = self.env.action_space()
        act.change_bus = {0}
        assert not act._cached_is_not_ambiguous
        
    def test_modif_int(self):
        act = self.env.action_space()
        act.set_bus = (0, 1)
        assert not act._cached_is_not_ambiguous
        
        act = self.env.action_space()
        act.set_bus = np.ones(self.env.dim_topo, dtype=int)
        assert not act._cached_is_not_ambiguous
        
        act = self.env.action_space()
        act.set_bus = [1 for _ in range(self.env.dim_topo)]
        assert not act._cached_is_not_ambiguous
        
        act = self.env.action_space()
        act.set_bus = [(0, 1)]
        assert not act._cached_is_not_ambiguous
        
        act = self.env.action_space()
        act.set_bus = {0: 1}
        assert not act._cached_is_not_ambiguous
        
    def test_modif_float(self):
        act = self.env.action_space()
        act.redispatch = (0, 1.)
        assert not act._cached_is_not_ambiguous
        
        act = self.env.action_space()
        act.redispatch = (0, np.nan)
        assert act._cached_is_not_ambiguous  # cache is not invalidated here (nothing has changed)
        
        act = self.env.action_space()
        act.redispatch = np.ones(self.env.n_gen, dtype=float)
        assert not act._cached_is_not_ambiguous
        
        act = self.env.action_space()
        act.redispatch = [1. for _ in range(self.env.n_gen)]
        assert not act._cached_is_not_ambiguous
        
        act = self.env.action_space()
        act.redispatch = [(0, 1.)]
        assert not act._cached_is_not_ambiguous
        
        act = self.env.action_space()
        act.set_bus = {0: 1.}
        assert not act._cached_is_not_ambiguous
        
    def test_modif_shunt(self):
        act = self.env.action_space({"shunt": {"shunt_p": [(0, 1.)]}})
        assert not act._cached_is_not_ambiguous
        act = self.env.action_space({"shunt": {"shunt_q": [(0, 1.)]}})
        assert not act._cached_is_not_ambiguous
        act = self.env.action_space({"shunt": {"shunt_bus": [(0, 1)]}})
        assert not act._cached_is_not_ambiguous
        act = self.env.action_space({"shunt": {"set_bus": [(0, 1)]}})
        assert not act._cached_is_not_ambiguous
        
    def test_modif_injection(self):
        act = self.env.action_space({"injection": {"load_p": [(0, 1.)]}})
        assert not act._cached_is_not_ambiguous
        act = self.env.action_space({"injection": {"load_q": [(0, 1.)]}})
        assert not act._cached_is_not_ambiguous
        act = self.env.action_space({"injection": {"prod_p": [(0, 1.)]}})
        assert not act._cached_is_not_ambiguous
        act = self.env.action_space({"injection": {"prod_v": [(0, 1.)]}})
        assert not act._cached_is_not_ambiguous
    
    def test_modif_redisp(self):
        act = self.env.action_space({"redispatch": [(0, 1.)]})
        assert not act._cached_is_not_ambiguous
        
    def test_modif_storage(self):
        act = self.env.action_space({"set_storage": [(0, 1.)]})
        assert not act._cached_is_not_ambiguous
        
    def test_modif_curtail(self):
        act = self.env.action_space({"curtail": [(0, 1.)]})
        assert not act._cached_is_not_ambiguous
        
    def test_modif_set_bus(self):
        act = self.env.action_space({"set_bus": [(0, 1.)]})
        assert not act._cached_is_not_ambiguous
        
    def test_modif_change_bus(self):
        act = self.env.action_space({"change_bus": [0]})
        assert not act._cached_is_not_ambiguous
        
    def test_modif_set_status(self):
        act = self.env.action_space({"set_line_status": [(0, 1)]})
        assert not act._cached_is_not_ambiguous
        
    def test_modif_hazards(self):
        act = self.env.action_space({"hazards": [0]})
        assert not act._cached_is_not_ambiguous
        
    def test_modif_maintenance(self):
        act = self.env.action_space({"maintenance": [0]})
        assert not act._cached_is_not_ambiguous
        
    def test_modif_change_status(self):
        act = self.env.action_space({"change_line_status": [0]})
        assert not act._cached_is_not_ambiguous
        
    def test_modif_detach_load(self):
        act = self.env.action_space({"detach_load": [0]})
        assert not act._cached_is_not_ambiguous
        
    def test_modif_detach_gen(self):
        act = self.env.action_space({"detach_gen": [0]})
        assert not act._cached_is_not_ambiguous
        
    def test_modif_detach_storage(self):
        act = self.env.action_space({"detach_storage": [0]})
        assert not act._cached_is_not_ambiguous
        
        
if __name__ == "__main__":
    unittest.main()
