# Copyright (c) 2025, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import unittest
import warnings

import grid2op
from grid2op.Action import CompleteAction
from grid2op.Backend import PandaPowerBackend
from grid2op.Space import AddDetailedTopoIEEE


class _PPBkForTestDetTopo(AddDetailedTopoIEEE, PandaPowerBackend):
    pass


class DetailedTopoDiscoTester_NoDetach(unittest.TestCase):
    def _aux_n_bb_per_sub(self):
        return 2
    
    def _aux_detach_is_allowed(self):
        return False
    
    def setUp(self) -> None:
        n_bb_per_sub = self._aux_n_bb_per_sub()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make(
                        "educ_case14_storage",
                        allow_detachment=self._aux_detach_is_allowed(),
                        n_busbar=n_bb_per_sub,
                        test=True,
                        backend=_PPBkForTestDetTopo(),
                        action_class=CompleteAction,
                        _add_to_name=f"{type(self).__name__}_{n_bb_per_sub}",
                    )
        params = self.env.parameters
        params.STOP_EP_IF_GEN_BREAK_CONSTRAINTS = False
        self.env.change_parameters(params)
        _ = self.env.reset(seed=0, options={"time serie id": 0})
        
    def tearDown(self) -> None:
        self.env.close()
        return super().tearDown()
    
    def test_disco_load_with_switch(self):
        # order of switches are:
        #  - load
        #  - gen
        #  - line or
        #  - line ex
        #  - storage
        #  - shunt
        
        # before that there are
        # the switch controlling the busbars
        nb_busbar = self._aux_n_bb_per_sub()
        start_id = (nb_busbar * (nb_busbar - 1) // 2) * type(self.env).n_sub
        
        # and for each element, the order is:
        # global switch (on / off for the element)
        # and then on switch per busbar
        load_id = 0
        load_0_deco_switch_id = start_id + load_id * (1 + nb_busbar) 
        act = self.env.action_space({"set_switch": [(load_0_deco_switch_id, -1)]})
        obs, reward, done, info = self.env.step(act)
        if not self._aux_detach_is_allowed():
            assert done
        else:
            assert not done
            assert obs.load_bus[load_id] == -1
            
    def test_disco_gen_with_switch(self):
        # order of switches are:
        #  - load
        #  - gen
        #  - line or
        #  - line ex
        #  - storage
        #  - shunt
        
        # before that there are
        # the switch controlling the busbars
        nb_busbar = self._aux_n_bb_per_sub()
        start_id = (nb_busbar * (nb_busbar - 1) // 2) * type(self.env).n_sub
        
        # and for each element, the order is:
        # global switch (on / off for the element)
        # and then on switch per busbar
        nb_load = type(self.env).n_load
        gen_id = 2  # gen id = 0 breaks
        gen_0_deco_switch_id = start_id + nb_load * (1 + nb_busbar) + gen_id * (1 + nb_busbar)
        act = self.env.action_space({"set_switch": [(gen_0_deco_switch_id, -1)]})
        obs, reward, done, info = self.env.step(act)
        if not self._aux_detach_is_allowed():
            assert done
        else:
            assert not done
            assert obs.gen_bus[gen_id] == -1
            
    def test_disco_line_or_with_switch(self):
        # order of switches are:
        #  - load
        #  - gen
        #  - line or
        #  - line ex
        #  - storage
        #  - shunt
        
        # before that there are
        # the switch controlling the busbars
        nb_busbar = self._aux_n_bb_per_sub()
        start_id = (nb_busbar * (nb_busbar - 1) // 2) * type(self.env).n_sub
        
        # and for each element, the order is:
        # global switch (on / off for the element)
        # and then on switch per busbar
        nb_load = type(self.env).n_load
        nb_gen = type(self.env).n_gen
        line_id = 2  # gen id = 0 breaks
        lor2_deco_switch_id = start_id + (nb_load + nb_gen) * (1 + nb_busbar) + line_id * (1 + nb_busbar)
        act = self.env.action_space({"set_switch": [(lor2_deco_switch_id, -1)]})
        obs, reward, done, info = self.env.step(act)
        assert not obs.line_status[line_id]
        assert obs.line_or_bus[line_id] == -1
        assert obs.line_ex_bus[line_id] == -1
        
    def test_disco_line_ex_with_switch(self):
        # order of switches are:
        #  - load
        #  - gen
        #  - line or
        #  - line ex
        #  - storage
        #  - shunt
        
        # before that there are
        # the switch controlling the busbars
        nb_busbar = self._aux_n_bb_per_sub()
        start_id = (nb_busbar * (nb_busbar - 1) // 2) * type(self.env).n_sub
        
        # and for each element, the order is:
        # global switch (on / off for the element)
        # and then on switch per busbar
        nb_load = type(self.env).n_load
        nb_gen = type(self.env).n_gen
        nb_line = type(self.env).n_line
        line_id = 3  # gen id = 0 breaks
        lex3_deco_switch_id = start_id + (nb_load + nb_gen + nb_line) * (1 + nb_busbar) + line_id * (1 + nb_busbar)
        act = self.env.action_space({"set_switch": [(lex3_deco_switch_id, -1)]})
        obs, reward, done, info = self.env.step(act)
        assert not obs.line_status[line_id]
        assert obs.line_or_bus[line_id] == -1
        assert obs.line_ex_bus[line_id] == -1
        
    
class DetailedTopoDiscoTester_Detach(DetailedTopoDiscoTester_NoDetach):
    def _aux_detach_is_allowed(self):
        return True
    