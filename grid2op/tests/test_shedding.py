# Copyright (c) 2024, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import warnings
import unittest
import numpy as np
import grid2op
from grid2op.Action import CompleteAction
from grid2op.Parameters import Parameters


class TestShedding(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()
        p = Parameters()
        p.MAX_SUB_CHANGED = 5
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make("rte_case5_example",
                                    param=p,
                                    action_class=CompleteAction,
                                    allow_detachment=True,
                                    test=True,
                                    _add_to_name=type(self).__name__)
        obs = self.env.reset(seed=0, options={"time serie id": "00"}) # Reproducibility
        self.load_lookup = {name:i for i,name in enumerate(self.env.name_load)}
        self.gen_lookup = {name:i for i,name in enumerate(self.env.name_gen)}

    def tearDown(self) -> None:
        self.env.close()

    def test_shedding_parameter_is_true(self):
        assert self.env._allow_detachment is True
        assert type(self.env).detachment_is_allowed
        assert type(self.env.backend).detachment_is_allowed
        assert self.env.backend.detachment_is_allowed

    def test_shed_single_load(self):
        # Check that a single load can be shed
        load_idx = self.load_lookup["load_4_2"]
        load_pos = self.env.load_pos_topo_vect[load_idx]
        act = self.env.action_space({
            "set_bus": [(load_pos, -1)]
        })
        obs, _, done, info = self.env.step(act)
        assert not done
        assert info["is_illegal"] is False
        assert obs.topo_vect[load_pos] == -1

    def test_shed_single_generator(self):
        # Check that a single generator can be shed
        gen_idx = self.gen_lookup["gen_0_0"]
        gen_pos = self.env.gen_pos_topo_vect[gen_idx]
        act = self.env.action_space({
            "set_bus": [(gen_pos, -1)]
        })
        obs, _, done, info = self.env.step(act)
        assert not done
        assert info["is_illegal"] is False
        assert obs.topo_vect[gen_pos] == -1

    def test_shed_multiple_loads(self):
        # Check that multiple loads can be shed at the same time
        load_idx1 = self.load_lookup["load_4_2"]
        load_idx2 = self.load_lookup["load_3_1"]
        load_pos1 = self.env.load_pos_topo_vect[load_idx1]
        load_pos2 = self.env.load_pos_topo_vect[load_idx2]
        act = self.env.action_space({
            "set_bus": [(load_pos1, -1), (load_pos2, -1)]
        })
        obs, _, done, info = self.env.step(act)
        assert not done
        assert info["is_illegal"] is False
        assert obs.topo_vect[load_pos1] == -1
        assert obs.topo_vect[load_pos2] == -1

    def test_shed_load_and_generator(self):
        # Check that load and generator can be shed at the same time
                # Check that multiple loads can be shed at the same time
        load_idx = self.load_lookup["load_4_2"]
        gen_idx = self.gen_lookup["gen_0_0"]
        load_pos = self.env.load_pos_topo_vect[load_idx]
        gen_pos = self.env.gen_pos_topo_vect[gen_idx]
        act = self.env.action_space({
            "set_bus": [(load_pos, -1), (gen_pos, -1)]
        })
        obs, _, done, info = self.env.step(act)
        assert not done
        assert info["is_illegal"] is False
        assert obs.topo_vect[load_pos] == -1
        assert obs.topo_vect[gen_pos] == -1

    def test_shedding_persistance(self):
        # Check that components remains disconnected if shed
        load_idx = self.load_lookup["load_4_2"]
        load_pos = self.env.load_pos_topo_vect[load_idx]
        act = self.env.action_space({
            "set_bus": [(load_pos, -1)]
        })
        _ = self.env.step(act)
        obs, _, done, _ = self.env.step(self.env.action_space({}))
        assert not done
        assert obs.topo_vect[load_pos] == -1

    def test_action_property(self):
        act = self.env.action_space()
        assert "detach_load" in type(act).authorized_keys
        act.detach_load = np.ones(act.n_load, dtype=bool)
        assert act._detach_load.all()
        
        act2 = self.env.action_space()
        act2.detach_load = 1
        assert act2._detach_load[1]
        
        act3 = self.env.action_space()
        act3.detach_load = [0, 2]
        assert act3._detach_load[0]
        assert act3._detach_load[2]
        
        for k, v in self.load_lookup.items():
            act4 = self.env.action_space()
            act4.detach_load = {k}
            assert act4._detach_load[v]

# TODO Shedding: test when backend does not support it is not set
# TODO shedding: test when user deactivates it it is not set

# TODO Shedding: Runner
# TODO Shedding: environment copied
# TODO Shedding: MultiMix environment
# TODO Shedding: TimedOutEnvironment
# TODO Shedding: MaskedEnvironment

if __name__ == "__main__":
    unittest.main()
