# Copyright (c) 2025, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.
import unittest
import warnings
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from getting_started import grid2op
from grid2op.Backend import PandaPowerBackend
from grid2op.Environment.EnvRecorder import EnvRecorder


class TestEnvRecorder(unittest.TestCase):

    @staticmethod
    def make_backend(detailed_infos_for_cascading_failures=False):
        return PandaPowerBackend(detailed_infos_for_cascading_failures)

    def test_recording(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = grid2op.make(
                "rte_case5_example",
                test=True,
                backend=self.make_backend(),
                _add_to_name=type(self).__name__
            )
            with TemporaryDirectory() as tmp_dir_name:
                tmp_dir_path = Path(tmp_dir_name)
                with EnvRecorder(env, tmp_dir_path, 3) as env_rec:
                    env_rec.reset()
                    do_nothing = env.action_space()
                    done = False
                    while not done:
                        _, _, done, _ = env_rec.step(do_nothing)

                for file_name in ['gen_p_before_curtail', 'gen_p', 'gen_v', 'load_p', 'load_q']:
                    pq_file = tmp_dir_path / f"{file_name}.parquet"
                    assert pq_file.is_file()

                gen_p_pq = pd.read_parquet(tmp_dir_path / "gen_p.parquet")
                assert gen_p_pq.shape == (95, 3)
                assert gen_p_pq.columns.tolist() == ['time', 'gen_0_0', 'gen_1_1']
