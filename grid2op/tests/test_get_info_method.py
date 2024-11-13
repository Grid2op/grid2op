# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import unittest
import warnings
from pathlib import Path
from grid2op.Environment import Environment, MultiMixEnvironment
from grid2op.tests.helper_path_test import *
import grid2op
import shutil

import pdb


class TestGetInfoMethod(unittest.TestCase):
    def setUp(self) -> None:
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make("educ_case14_storage", test=True, _add_to_name=type(self).__name__)
            
    def tearDown(self) -> None:
        self.env.close()
        return super().tearDown()
    
    def test_get_line(self):
        env_cls = type(self.env)
        # raises error when input ill-formated
        with self.assertRaises(RuntimeError):
            env_cls.get_line_info()
        with self.assertRaises(TypeError):
            env_cls.get_line_info("0_1_0")
        with self.assertRaises(TypeError):
            env_cls.get_line_info(0)
        with self.assertRaises(RuntimeError):
            env_cls.get_line_info(line_name="0_1_0", line_id=0)
        # raises error because of non existing object
        with self.assertRaises(RuntimeError):
            env_cls.get_line_info(line_name="unknown_line")
        with self.assertRaises(RuntimeError):
            env_cls.get_line_info(line_id=-1)
        with self.assertRaises(RuntimeError):
            env_cls.get_line_info(line_id=env_cls.n_line)
        
        # return correct result
        for line_id, line_name, sub_or, sub_ex in zip(range(env_cls.n_line),
                                                      env_cls.name_line,
                                                      env_cls.line_or_to_subid,
                                                      env_cls.line_ex_to_subid):
            line_id_res, line_name_res, sub_or_res, sub_ex_res = env_cls.get_line_info(line_id=line_id)
            assert line_id == line_id_res, f"{line_id_res} vs {line_id}"
            assert line_name == line_name_res, f"{line_name_res} vs {line_name}"
            assert sub_or_res == sub_or, f"{sub_or_res} vs {sub_or}"
            assert sub_ex_res == sub_ex, f"{sub_ex_res} vs {sub_ex}"
            
        for line_id, line_name, sub_or, sub_ex in zip(range(env_cls.n_line),
                                                      env_cls.name_line,
                                                      env_cls.line_or_to_subid,
                                                      env_cls.line_ex_to_subid):
            line_id_res, line_name_res, sub_or_res, sub_ex_res = env_cls.get_line_info(line_name=line_name)
            assert line_id == line_id_res, f"{line_id_res} vs {line_id}"
            assert line_name == line_name_res, f"{line_name_res} vs {line_name}"
            assert sub_or_res == sub_or, f"{sub_or_res} vs {sub_or}"
            assert sub_ex_res == sub_ex, f"{sub_ex_res} vs {sub_ex}"
        
    def test_get_load(self):
        env_cls = type(self.env)
        # raises error when input ill-formated
        with self.assertRaises(RuntimeError):
            env_cls.get_load_info()
        with self.assertRaises(TypeError):
            env_cls.get_load_info("load_1_0")
        with self.assertRaises(TypeError):
            env_cls.get_load_info(0)
        with self.assertRaises(RuntimeError):
            env_cls.get_load_info(load_name="load_1_0", load_id=0)
        # raises error because of non existing object
        with self.assertRaises(RuntimeError):
            env_cls.get_load_info(load_name="unknown_load")
        with self.assertRaises(RuntimeError):
            env_cls.get_load_info(load_id=-1)
        with self.assertRaises(RuntimeError):
            env_cls.get_load_info(load_id=env_cls.n_load)
        
        # return correct result
        for el_id, el_name, sub_id in zip(range(env_cls.n_load),
                                          env_cls.name_load,
                                          env_cls.load_to_subid):
            el_id_res, el_name_res, sub_res = env_cls.get_load_info(load_id=el_id)
            assert el_id == el_id_res, f"{el_id_res} vs {el_id}"
            assert el_name == el_name_res, f"{el_name_res} vs {el_name}"
            assert sub_res == sub_id, f"{sub_res} vs {sub_id}"
            
        for el_id, el_name, sub_id in zip(range(env_cls.n_load),
                                          env_cls.name_load,
                                          env_cls.load_to_subid):
            el_id_res, el_name_res, sub_res = env_cls.get_load_info(load_name=el_name)
            assert el_id == el_id_res, f"{el_id_res} vs {el_id}"
            assert el_name == el_name_res, f"{el_name_res} vs {el_name}"
            assert sub_res == sub_id, f"{sub_res} vs {sub_id}"
            
    def test_get_gen(self):
        env_cls = type(self.env)
        # raises error when input ill-formated
        with self.assertRaises(RuntimeError):
            env_cls.get_gen_info()
        with self.assertRaises(TypeError):
            env_cls.get_gen_info("gen_5_2")
        with self.assertRaises(TypeError):
            env_cls.get_gen_info(0)
        with self.assertRaises(RuntimeError):
            env_cls.get_gen_info(gen_name="gen_5_2", gen_id=0)
        # raises error because of non existing object
        with self.assertRaises(RuntimeError):
            env_cls.get_gen_info(gen_name="unknown_gen")
        with self.assertRaises(RuntimeError):
            env_cls.get_gen_info(gen_id=-1)
        with self.assertRaises(RuntimeError):
            env_cls.get_gen_info(gen_id=env_cls.n_gen)
        
        # return correct result
        for el_id, el_name, sub_id in zip(range(env_cls.n_gen),
                                          env_cls.name_gen,
                                          env_cls.gen_to_subid):
            el_id_res, el_name_res, sub_res = env_cls.get_gen_info(gen_id=el_id)
            assert el_id == el_id_res, f"{el_id_res} vs {el_id}"
            assert el_name == el_name_res, f"{el_name_res} vs {el_name}"
            assert sub_res == sub_id, f"{sub_res} vs {sub_id}"
            
        for el_id, el_name, sub_id in zip(range(env_cls.n_gen),
                                          env_cls.name_gen,
                                          env_cls.gen_to_subid):
            el_id_res, el_name_res, sub_res = env_cls.get_gen_info(gen_name=el_name)
            assert el_id == el_id_res, f"{el_id_res} vs {el_id}"
            assert el_name == el_name_res, f"{el_name_res} vs {el_name}"
            assert sub_res == sub_id, f"{sub_res} vs {sub_id}"
            
    def test_get_storage(self):
        env_cls = type(self.env)
        # raises error when input ill-formated
        with self.assertRaises(RuntimeError):
            env_cls.get_storage_info()
        with self.assertRaises(TypeError):
            env_cls.get_storage_info("storage_5_0")
        with self.assertRaises(TypeError):
            env_cls.get_storage_info(0)
        with self.assertRaises(RuntimeError):
            env_cls.get_storage_info(storage_name="storage_5_0", storage_id=0)
        # raises error because of non existing object
        with self.assertRaises(RuntimeError):
            env_cls.get_storage_info(storage_name="unknown_storage")
        with self.assertRaises(RuntimeError):
            env_cls.get_storage_info(storage_id=-1)
        with self.assertRaises(RuntimeError):
            env_cls.get_storage_info(storage_id=env_cls.n_storage)
        
        # return correct result
        for el_id, el_name, sub_id in zip(range(env_cls.n_storage),
                                          env_cls.name_storage,
                                          env_cls.storage_to_subid):
            el_id_res, el_name_res, sub_res = env_cls.get_storage_info(storage_id=el_id)
            assert el_id == el_id_res, f"{el_id_res} vs {el_id}"
            assert el_name == el_name_res, f"{el_name_res} vs {el_name}"
            assert sub_res == sub_id, f"{sub_res} vs {sub_id}"
            
        for el_id, el_name, sub_id in zip(range(env_cls.n_storage),
                                          env_cls.name_storage,
                                          env_cls.storage_to_subid):
            el_id_res, el_name_res, sub_res = env_cls.get_storage_info(storage_name=el_name)
            assert el_id == el_id_res, f"{el_id_res} vs {el_id}"
            assert el_name == el_name_res, f"{el_name_res} vs {el_name}"
            assert sub_res == sub_id, f"{sub_res} vs {sub_id}"
        
        
if __name__ == "__main__":
    unittest.main()
