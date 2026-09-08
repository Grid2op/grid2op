# Copyright (c) 2025, RTE (https://www.rte-france.com)
# See AUTHORS.txt and https://github.com/Grid2Op/grid2op/pull/319
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.


import warnings
import grid2op
import numpy as np
import unittest

from grid2op.Observation import CompleteObservation


class TestIssue731(unittest.TestCase):
    def setUp(self):
        env_name = "l2rpn_wcci_2022"
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make(env_name,
                                    test=True,
                                    observation_class=CompleteObservation,
                                    _add_to_name=f"{type(self).__name__}")
        self.obs = self.env.reset(seed=0, options={"time serie id": 0})
        return super().setUp()
    def tearDown(self):
        self.env.close()
        return super().tearDown()
    
    def test_issue(self):    
        assert np.allclose(self.obs.gen_p_delta, self.env.observation_space.from_vect(self.obs.to_vect()).gen_p_delta) 
