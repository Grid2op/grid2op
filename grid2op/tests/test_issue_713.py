# Copyright (c) 2025, RTE (https://www.rte-france.com)
# See AUTHORS.txt and https://github.com/Grid2Op/grid2op/pull/319
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import warnings
import grid2op
from grid2op.Action import BaseAction
import unittest


class TestIssue713(unittest.TestCase):
    def setUp(self) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make("educ_case14_storage",
                                    test=True,
                                    action_class=BaseAction,
                                    _add_to_name=type(self).__name__)
        self.init_obs = self.env.reset(seed=0, options={"time serie id":0})
        return super().setUp()
    
    def test_proper_date_time_after_reset(self):
        for_env = self.init_obs.get_forecast_env()
        for_obs = for_env.reset()
        self.assertEqual(for_obs.hour_of_day, 0)
        self.assertEqual(for_obs.minute_of_hour, 0)
        self.assertEqual(for_obs.year, 2019)
        self.assertEqual(for_obs.day, 12)
        self.assertEqual(for_obs.day_of_week, 5)
        
    def test_proper_date_time_after_step(self):
        for_env = self.init_obs.get_forecast_env()
        for_obs = for_env.reset()
        for_obs, *_ = for_env.step(self.env.action_space())
        self.assertEqual(for_obs.hour_of_day, 0)
        self.assertEqual(for_obs.minute_of_hour, 5)
        self.assertEqual(for_obs.year, 2019)
        self.assertEqual(for_obs.day, 12)
        self.assertEqual(for_obs.day_of_week, 5)
        
    def test_proper_date_time_wuithout_reset(self):
        for_env = self.init_obs.get_forecast_env()
        for_obs = for_env.get_obs()
        self.assertEqual(for_obs.hour_of_day, 0)
        self.assertEqual(for_obs.minute_of_hour, 0)
        self.assertEqual(for_obs.year, 2019)
        self.assertEqual(for_obs.day, 12)
        self.assertEqual(for_obs.day_of_week, 5)
        
        for_obs, *_ = for_env.step(self.env.action_space())
        self.assertEqual(for_obs.hour_of_day, 0)
        self.assertEqual(for_obs.minute_of_hour, 5)
        self.assertEqual(for_obs.year, 2019)
        self.assertEqual(for_obs.day, 12)
        self.assertEqual(for_obs.day_of_week, 5)
        
        
if __name__ == "__main__":
    unittest.main()
