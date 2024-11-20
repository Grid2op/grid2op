# Copyright (c) 2024, RTE (https://www.rte-france.com)
# See AUTHORS.txt and https://github.com/Grid2Op/grid2op/pull/319
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

from logging import Logger
import unittest
import warnings

try: # Grid2Op available locally
    from .helper_path_test import PATH_DATA_TEST
except: # Grid2OP available as a package
    from helper_path_test import PATH_DATA_TEST
import grid2op
from grid2op.Exceptions import ChronicsError, EnvError
from grid2op.Action import BaseAction
from grid2op.Environment import BaseEnv
from grid2op.Reward import BaseReward


class WeirdReward(BaseReward):
    def __init__(self, logger: Logger = None):
        super().__init__(logger)
        
    def __call__(self, action: BaseAction, env:BaseEnv, has_error: bool, is_done: bool, is_illegal: bool, is_ambiguous: bool) -> float:
        return len(env.chronics_handler.get_name())
    
    
class Issue657Tester(unittest.TestCase):
    def setUp(self):
        self.env_name = "l2rpn_case14_sandbox"
        # create first env
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make("l2rpn_case14_sandbox", test=True, reward_class=WeirdReward)
            
    def tearDown(self) -> None:
        self.env.close()
        return super().tearDown()
    
    def test_issue_657(self):
        obs = self.env.reset()
        obs.simulate(self.env.action_space())
        self.env.step(self.env.action_space())

    def test_broader_names(self):
        obs = self.env.reset()
        obs_ch = obs._obs_env.chronics_handler
        for attr_nm in self.env.chronics_handler.__dict__:
            try:
                getattr(obs_ch, attr_nm)
            except (EnvError, ChronicsError) as exc_:
                # access to some attributes / function might return these type of errors
                pass 
            except AttributeError as exc_:
                raise TypeError(f"No know attribute {attr_nm} for obs_chronics_handler") from exc_
            
        for attr_nm in self.env.chronics_handler.real_data.__dict__:
            try:
                getattr(obs_ch, attr_nm)
            except (EnvError, ChronicsError) as exc_:
                # access to some attributes / function might return these type of errors
                pass 
            except AttributeError as exc_:
                raise TypeError(f"No know attribute {attr_nm} (from real_data / GridValue) for obs_chronics_handler") from exc_
            
            
if __name__ == "__main__":
    unittest.main()
