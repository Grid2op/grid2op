# Copyright (c) 2023, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

# %%
import os
import grid2op
from grid2op.Chronics import MultifolderWithCache
import unittest
import warnings


class TestMultifolderWithCacheSeed(unittest.TestCase):
    def test_box_action_space(self):
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = grid2op.make("l2rpn_case14_sandbox", test=True, _add_to_name=type(self).__name__,
                               chronics_class=MultifolderWithCache)
        
        # I take the last chronics on purpose so that its index is larger than the number of selected chronics (only one here)
        env.chronics_handler.real_data.set_filter(lambda x: os.path.basename(x) == "0002")
        env.chronics_handler.reset()

        try:
            env.reset(seed=0)
        except Exception as e:
            self.fail(f"{type(self).__name__} raised an exception: {e}")

        
        
# %%        
if __name__ == "__main__":
    unittest.main()