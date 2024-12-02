# Copyright (c) 2024, RTE (https://www.rte-france.com)
# See AUTHORS.txt and https://github.com/Grid2Op/grid2op/pull/319
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import numpy as np
from logging import Logger
import unittest
import warnings


from helper_path_test import PATH_DATA_TEST
import grid2op
from grid2op.dtypes import dt_int, dt_float
from grid2op.gym_compat import BoxGymObsSpace
from grid2op.gym_compat.utils import _compute_extra_power_for_losses
from grid2op.Exceptions import ChronicsError, EnvError
    
    
class Issue665Tester(unittest.TestCase):
    def setUp(self):
        self.env_name = "l2rpn_idf_2023"
        # create first env
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make("l2rpn_idf_2023", test=True)
        self.dict_properties = BoxGymObsSpace(self.env.observation_space)._dict_properties
            
    def tearDown(self) -> None:
        self.env.close()
        return super().tearDown()
    
    def test_issue_665(self):
        attributes_names = set(self.dict_properties.keys())
        attr_with_a_problem = set() # I put an attribute here if at least one bound has been exceeded at least once
        attr_without_a_problem = set(self.dict_properties.keys()) # I remove an attribute from here if at least one bound has been exceeded at least once

        i = 0
        while i < 5 and not attr_without_a_problem:
            obs = self.env.reset()
            obs_temp = self.env.observation_space._template_obj
            
            # I check only attributes which has not exceeded their bounds yet
            for attr_name in attr_without_a_problem:
                attr = getattr(obs_temp, attr_name)
                low = self.dict_properties[attr_name][0]
                high = self.dict_properties[attr_name][1]

                ids = np.where((attr < low) | (attr > high))[0]
                if ids.shape[0] > 0: # Case where at least a bound has been exceeded
                    # I uppdate my set
                    attr_with_a_problem.add(attr_name)
                    # I print a value (the one with the lower index) that exceeded its bounds
                    id0 = ids[0]
                    print(f"The {attr_name} attribute is out of the bounds with index {id0}. Bounds : {low[id0]} <= {high[id0]}, value: {attr[id0]}.")

            # I uppdate my set
            attr_without_a_problem = attributes_names - attr_with_a_problem
            i+=1

        assert not attr_with_a_problem

if __name__ == "__main__":
    unittest.main()
