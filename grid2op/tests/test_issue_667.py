# Copyright (c) 2024, RTE (https://www.rte-france.com)
# See AUTHORS.txt and https://github.com/Grid2Op/grid2op/pull/319
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import json
import tempfile
import warnings
import grid2op
from grid2op.Action import BaseAction
from pathlib import Path
import unittest


class TestIssue667(unittest.TestCase):
    def setUp(self) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make("l2rpn_wcci_2022",  # we need t least 2 shunts and 2 storage units
                                    test=True,
                                    action_class=BaseAction,
                                    _add_to_name=type(self).__name__)
        self.init_obs = self.env.reset(seed=0, options={"time serie id":0})
        return super().setUp()
    
    def test_set_line_status(self):
        all_but_one_lines_on = self.env.action_space({"set_line_status":[(name, 1) for name in self.env.name_line[0:-1]]})
        all_lines_on = self.env.action_space({"set_line_status":[(name, 1) for name in self.env.name_line[:]]})

        with tempfile.TemporaryDirectory() as tmpdirname:
            tmp_path = Path(tmpdirname) 
            for act in [all_but_one_lines_on, all_lines_on]:
                with open(tmp_path/ "act.json", "w") as f:
                    json.dump(act.as_serializable_dict(), f)
                with open(tmp_path / "act.json", "r") as f:
                    dict_ = json.load(f)
                act2 = self.env.action_space(dict_)
                assert act == act2
    
    def test_change_line_status(self):
        all_but_one_lines_on = self.env.action_space({"change_line_status":[name for name in self.env.name_line[0:-1]]})
        all_lines_on = self.env.action_space({"change_line_status":[name for name in self.env.name_line[:]]})

        with tempfile.TemporaryDirectory() as tmpdirname:
            tmp_path = Path(tmpdirname) 
            for act in [all_but_one_lines_on, all_lines_on]:
                with open(tmp_path/ "act.json", "w") as f:
                    json.dump(act.as_serializable_dict(), f)
                with open(tmp_path / "act.json", "r") as f:
                    dict_ = json.load(f)
                act2 = self.env.action_space(dict_)
                assert act == act2
        
    def test_redispatching(self):
        all_gens_but_one = self.env.action_space({"redispatch":[(name, 0.01) for name in self.env.name_gen[0:-1]]})
        all_gens = self.env.action_space({"redispatch":[(name, 0.01) for name in self.env.name_gen[:]]})

        with tempfile.TemporaryDirectory() as tmpdirname:
            for act in [all_gens_but_one, all_gens]:
                with open(Path(tmpdirname) / "act.json", "w") as f:
                    json.dump(act.as_serializable_dict(), f)
                with open(Path(tmpdirname) / "act.json", "r") as f:
                    dict_ = json.load(f)
                act2 = self.env.action_space(dict_)
                assert act == act2
        
    def test_curtail(self):
        all_gens_but_one = self.env.action_space({"curtail":[(name, 0.01) for name in self.env.name_gen[0:-1]]})
        all_gens = self.env.action_space({"curtail":[(name, 0.01) for name in self.env.name_gen[:]]})

        with tempfile.TemporaryDirectory() as tmpdirname:
            for act in [all_gens_but_one, all_gens]:
                with open(Path(tmpdirname) / "act.json", "w") as f:
                    json.dump(act.as_serializable_dict(), f)
                with open(Path(tmpdirname) / "act.json", "r") as f:
                    dict_ = json.load(f)
                act2 = self.env.action_space(dict_)
                assert act == act2
        
    def test_storage(self):
        all_gens_but_one = self.env.action_space({"set_storage":[(name, 0.01) for name in self.env.name_storage[0:-1]]})
        all_gens = self.env.action_space({"set_storage":[(name, 0.01) for name in self.env.name_storage[:]]})

        with tempfile.TemporaryDirectory() as tmpdirname:
            for act in [all_gens_but_one, all_gens]:
                with open(Path(tmpdirname) / "act.json", "w") as f:
                    json.dump(act.as_serializable_dict(), f)
                with open(Path(tmpdirname) / "act.json", "r") as f:
                    dict_ = json.load(f)
                act2 = self.env.action_space(dict_)
                assert act == act2
                
    def test_set_bus(self):
        for el_type, el_vect in zip(["loads_id", "generators_id", "lines_or_id", "lines_ex_id", "storages_id"],
                                    [self.env.name_load, self.env.name_gen, self.env.name_line, self.env.name_line, self.env.name_storage]):
            all_but_one_lines_on = self.env.action_space({"set_bus": {el_type: [(name, 2) for name in el_vect[0:-1]]}})
            all_lines_on = self.env.action_space({"set_bus": {el_type: [(name, 1) for name in el_vect[:]]}})

            with tempfile.TemporaryDirectory() as tmpdirname:
                tmp_path = Path(tmpdirname) 
                for act in [all_but_one_lines_on, all_lines_on]:
                    with open(tmp_path/ "act.json", "w") as f:
                        json.dump(act.as_serializable_dict(), f)
                    with open(tmp_path / "act.json", "r") as f:
                        dict_ = json.load(f)
                    act2 = self.env.action_space(dict_)
                    assert act == act2
                
    def test_change_bus(self):
        for el_type, el_vect in zip(["loads_id", "generators_id", "lines_or_id", "lines_ex_id", "storages_id"],
                                    [self.env.name_load, self.env.name_gen, self.env.name_line, self.env.name_line, self.env.name_storage]):
            all_but_one_lines_on = self.env.action_space({"change_bus": {el_type: [name for name in el_vect[0:-1]]}})
            all_lines_on = self.env.action_space({"change_bus": {el_type: [name for name in el_vect[:]]}})

            with tempfile.TemporaryDirectory() as tmpdirname:
                tmp_path = Path(tmpdirname) 
                for act in [all_but_one_lines_on, all_lines_on]:
                    with open(tmp_path/ "act.json", "w") as f:
                        json.dump(act.as_serializable_dict(), f)
                    with open(tmp_path / "act.json", "r") as f:
                        dict_ = json.load(f)
                    act2 = self.env.action_space(dict_)
                    assert act == act2
                
    def test_injection(self):
        for el_type, el_vect in zip(["prod_p", "prod_v", "load_p", "load_q"],
                                    [self.env.name_gen, self.env.name_gen, self.env.name_load, self.env.name_load]):
            all_but_one_lines_on = self.env.action_space({"injection": {el_type: {name: 1.0 for name in el_vect[0:-1]}}})
            all_lines_on = self.env.action_space({"injection": {el_type: {name: 2. for name in el_vect[:]}}})

            with tempfile.TemporaryDirectory() as tmpdirname:
                tmp_path = Path(tmpdirname) 
                for act in [all_but_one_lines_on, all_lines_on]:
                    with open(tmp_path/ "act.json", "w") as f:
                        json.dump(act.as_serializable_dict(), f)
                    with open(tmp_path / "act.json", "r") as f:
                        dict_ = json.load(f)
                    act2 = self.env.action_space(dict_)
                    assert act == act2
                
    def test_shunt(self):
        for el_type, prop_nm in zip(["set_bus", "shunt_p", "shunt_q", "shunt_bus"],
                                    ["_shunt_bus", "_shunt_p", "_shunt_q", "_shunt_bus"]):
            all_but_one_shunt = self.env.action_space({"shunt": {el_type: {name: 1 for name in self.env.name_shunt[0:-1]}}})
            assert all_but_one_shunt._modif_shunt
            all_shunts_bus2 = self.env.action_space({"shunt": {el_type: {name: 2 for name in self.env.name_shunt[:]}}})
            assert all_shunts_bus2._modif_shunt

            with tempfile.TemporaryDirectory() as tmpdirname:
                tmp_path = Path(tmpdirname) 
                for act in [all_but_one_shunt, all_shunts_bus2]:
                    with open(tmp_path/ "act.json", "w") as f:
                        json.dump(act.as_serializable_dict(), f)
                    with open(tmp_path / "act.json", "r") as f:
                        dict_ = json.load(f)
                    act2 = self.env.action_space(dict_)
                    assert act == act2
    
    
if __name__ == "__main__":
    unittest.main()
