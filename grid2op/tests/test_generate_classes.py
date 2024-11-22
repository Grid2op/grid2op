# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import re
import unittest
import warnings
from pathlib import Path
from grid2op.Environment import Environment, MultiMixEnvironment
from grid2op.tests.helper_path_test import *
import grid2op
from grid2op.Space import GRID2OP_CLASSES_ENV_FOLDER
import shutil

import pdb


class TestGenerateFile(unittest.TestCase):
    def _aux_assert_exists_then_delete(self, env):
        path = Path(env.get_path_env()) / GRID2OP_CLASSES_ENV_FOLDER
        assert path.exists(), f"path {path} does not exists"
        shutil.rmtree(path, ignore_errors=True)
        return
        if isinstance(env, MultiMixEnvironment):
            # for mix in env:
                # self._aux_assert_exists_then_delete(mix)
            self._aux_assert_exists_then_delete(env.mix_envs[env.all_names[0]])
        elif isinstance(env, Environment):
            path = Path(env.get_path_env()) / GRID2OP_CLASSES_ENV_FOLDER
            assert path.exists(), f"path {path} does not exists"
            shutil.rmtree(path, ignore_errors=True)
        else:
            raise RuntimeError("Unknown env type")
        
    def list_env(self):
        env_with_alert = os.path.join(
            PATH_DATA_TEST, "l2rpn_idf_2023_with_alert"
        )
        return grid2op.list_available_test_env() + [env_with_alert]
    
    def test_can_generate(self):
        for env_nm in self.list_env():
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                    env = grid2op.make(env_nm, test=True, _add_to_name=type(self).__name__+"test_generate")
                env.generate_classes()
            finally:
                self._aux_assert_exists_then_delete(env)
                env.close()
        
    def test_can_load(self):
        _add_to_name = type(self).__name__+"test_load"
        for env_nm in self.list_env():
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                env = grid2op.make(env_nm,
                                   test=True,
                                   _add_to_name=_add_to_name)
            env.generate_classes()
            cls_nm_tmp = f"PandaPowerBackend{_add_to_name}"
            cls_nm_end = f"{cls_nm_tmp}$"
            cls_nm_twice = f"{cls_nm_tmp}.+{cls_nm_end}"
            assert re.search(cls_nm_end, type(env).__name__) is not None # name of the backend and "add_to_name" should appear once
            assert re.search(cls_nm_twice, type(env).__name__) is None  # they should not appear twice !
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                try:
                    env2 = grid2op.make(env_nm,
                                        test=True,
                                        experimental_read_from_local_dir=True,
                                        _add_to_name=_add_to_name)
                    env2.close()
                except RuntimeError as exc_:
                    raise RuntimeError(f"Error for {env_nm}") from exc_
            self._aux_assert_exists_then_delete(env)
            env.close()
        
        
if __name__ == "__main__":
    unittest.main()
