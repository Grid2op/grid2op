# Copyright (c) 2025, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.
import json
import unittest
import warnings
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

import grid2op
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

                # check all files have been generated
                for file_name in ['gen_detached.parquet',
                                  'line_ex_q.parquet',
                                  'storage.parquet',
                                  'storage_detached.parquet',
                                  'line_rho.parquet',
                                  'gen_p_before_curtail.parquet',
                                  'line_or_bus.parquet',
                                  'line_ex_theta.parquet',
                                  'load_q.parquet',
                                  'line_ex_a.parquet',
                                  'line_or_v.parquet',
                                  'line_or_q.parquet',
                                  'line_or_theta.parquet',
                                  'line_or_p.parquet',
                                  'storage_power.parquet',
                                  'load_p.parquet',
                                  'gen_theta.parquet',
                                  'line_ex_v.parquet',
                                  'shunt_bus.parquet',
                                  'line_ex_p.parquet',
                                  'storage_p_detached.parquet',
                                  'gen_bus.parquet',
                                  'line_thermal_limit.parquet',
                                  'load_p_detached.parquet',
                                  'gen_actual_dispatch.parquet',
                                  'shunt_q.parquet',
                                  'line_or_a.parquet',
                                  'gen_q.parquet',
                                  'storage_theta.parquet',
                                  'gen.parquet',
                                  'load_v.parquet',
                                  'gen_p.parquet',
                                  'load.parquet',
                                  'storage_power_target.parquet',
                                  'load_q_detached.parquet',
                                  'shunt_v.parquet',
                                  'line_status.parquet',
                                  'gen_target_dispatch.parquet',
                                  'shunt.parquet',
                                  'storage_charge.parquet',
                                  'env.json',
                                  'line.parquet',
                                  'load_theta.parquet',
                                  'storage_bus.parquet',
                                  'load_bus.parquet',
                                  'gen_v.parquet',
                                  'line_ex_bus.parquet',
                                  'gen_p_detached.parquet',
                                  'shunt_p.parquet',
                                  'actions.parquet']:
                    pq_file = tmp_dir_path / f"{file_name}"
                    assert pq_file.is_file()

                # check one of the table file content
                gen_p_pq = pd.read_parquet(tmp_dir_path / "gen_p.parquet")
                assert gen_p_pq.shape == (96, 3)
                assert gen_p_pq.columns.tolist() == ['time', 'gen_0_0', 'gen_1_1']

                # check the environment infos file content
                with open(tmp_dir_path / "env.json", "r", encoding="utf-8") as f:
                    env_infos = json.load(f)
                    assert env_infos['grid2op_version']
                    assert env_infos['name'] == 'rte_case5_examplePandaPowerBackendTestEnvRecorder'
                    assert env_infos['path']
                    assert env_infos['backend'] == 'PandaPowerBackend_rte_case5_examplePandaPowerBackendTestEnvRecorder'
                    assert env_infos['n_sub'] == 5
                    assert env_infos['n_busbar_per_sub'] == 2
