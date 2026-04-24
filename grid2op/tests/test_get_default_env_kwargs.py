# Copyright (c) 2026, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import json
import os
import shutil
import tempfile
import unittest
import warnings

from grid2op.Action.baseAction import BaseAction
from grid2op.Chronics.chronicsHandler import ChronicsHandler
from grid2op.Chronics.gridValue import GridValue
from grid2op.Exceptions.envExceptions import EnvError
from grid2op.MakeEnv._aux_var import TEST_DEV_ENVS
from grid2op.Opponent.baseActionBudget import BaseActionBudget
from grid2op.Opponent.baseOpponent import BaseOpponent
from grid2op.Opponent.opponentSpace import OpponentSpace
from grid2op.Reward.baseReward import BaseReward
from grid2op.Parameters import Parameters
from grid2op.Backend import PandaPowerBackend
from grid2op.MakeEnv.get_default_env_kwargs import (
    get_default_env_kwargs,
    ERR_MSG_KWARGS,    
    )
from grid2op.Observation import CompleteObservation
from grid2op.Rules.BaseRules import BaseRules
from grid2op.VoltageControler import ControlVoltageFromFile
from grid2op.operator_attention.attention_budget import LinearAttentionBudget

mappings = {
    "backend": PandaPowerBackend,
    "observation_class": CompleteObservation,
    "param": Parameters(),
    "gamerules_class": BaseRules,
    "reward_class": BaseReward,
    "action_class": BaseAction,
    "data_feeding_kwargs": dict(),
    "chronics_class": GridValue,
    "chronics_handler": ChronicsHandler,
    "voltagecontroler_class":ControlVoltageFromFile,
    "names_chronics_to_grid": dict(),
    "other_rewards": dict(),
    "chronics_path": "",
    "grid_path": "",
    "opponent_space_type": OpponentSpace,
    "opponent_action_class": BaseAction,
    "opponent_class": BaseOpponent,
    "opponent_attack_duration": 0,
    "opponent_attack_cooldown": 0,
    "opponent_init_budget": 0.,
    "opponent_budget_class": BaseActionBudget,
    "opponent_budget_per_ts": 0.,
    "kwargs_opponent": dict(),
    "has_attention_budget": True,
    "attention_budget_class": LinearAttentionBudget,
    "kwargs_attention_budget": dict(),
    "difficulty": "",
    "kwargs_observation": dict(),
    "observation_backend_class": CompleteObservation,
    "observation_backend_kwargs": dict(),
    "class_in_file": True,
}


class _MyBack(PandaPowerBackend):
    IS_MY_BACKEND = True


class _MyObs(CompleteObservation):
    IS_MY_OBS = True
        
        
class TestGetDefaultEnvKwargs(unittest.TestCase):
    def call_get_def_kwargs(self, dataset_path=None, **kwargs):
        return get_default_env_kwargs(
            dataset_path=TEST_DEV_ENVS["l2rpn_case14_sandbox"] if dataset_path is None else dataset_path,
            logger=None,
            n_busbar=2,
            allow_detachment=False,
            _add_cls_nm_bk=True,
            _add_to_name="",
            _compat_glop_version=None,
            _overload_name_multimix=None,
            **kwargs
        )
        
    def test_backend(self):        
        # test correctly set from kwargs
        res = self.call_get_def_kwargs(backend=_MyBack())
        assert isinstance(res[0]["backend"], _MyBack)
        
        # test correctly set from config
        with tempfile.TemporaryDirectory() as tmp:
            shutil.copy(
                f'{TEST_DEV_ENVS["l2rpn_case14_sandbox"]}/grid.json',
                os.path.join(tmp, "grid.json")
                )
            os.mkdir(os.path.join(tmp, "chronics"))
            
            with open(os.path.join(tmp, "config.py"), "w") as f:
                f.write("from grid2op.tests.test_get_default_env_kwargs import _MyBack\n")
                f.write("config ={'backend': _MyBack()}\n")
            res = self.call_get_def_kwargs(tmp)  
            assert hasattr(type(res[0]["backend"]), "IS_MY_BACKEND")
            assert type(res[0]["backend"]).IS_MY_BACKEND
        
        # test correctly set from config, using backend_class
        with tempfile.TemporaryDirectory() as tmp:
            shutil.copy(
                f'{TEST_DEV_ENVS["l2rpn_case14_sandbox"]}/grid.json',
                os.path.join(tmp, "grid.json")
                )
            os.mkdir(os.path.join(tmp, "chronics"))
            with open(os.path.join(tmp, "config.py"), "w") as f:
                f.write("from grid2op.tests.test_get_default_env_kwargs import _MyBack\n")
                f.write("config ={'backend_class': _MyBack}\n")
            res = self.call_get_def_kwargs(tmp)  
            assert hasattr(type(res[0]["backend"]), "IS_MY_BACKEND")
            assert type(res[0]["backend"]).IS_MY_BACKEND
        
    def test_chronics_path(self):
        # test correctly set from kwargs
        res = self.call_get_def_kwargs(chronics_path=TEST_DEV_ENVS["l2rpn_icaps_2021"])
        assert res[3].path == TEST_DEV_ENVS["l2rpn_icaps_2021"]
        
        # test correctly set from config
        with tempfile.TemporaryDirectory() as tmp:
            shutil.copy(
                f'{TEST_DEV_ENVS["l2rpn_case14_sandbox"]}/grid.json',
                os.path.join(tmp, "grid.json")
                )
            
            with open(os.path.join(tmp, "config.py"), "w") as f:
                f.write("from grid2op.tests.test_get_default_env_kwargs import _MyBack\n")
                f.write(f"config ={{'chronics_path': \"{TEST_DEV_ENVS['l2rpn_icaps_2021']}\" }}\n")
            res = self.call_get_def_kwargs(tmp)  
            assert res[3].path == TEST_DEV_ENVS["l2rpn_icaps_2021"]
            
    def test_names_chronics_to_grid(self):
        tmp_ = {"2": "1"}
        res = self.call_get_def_kwargs(names_chronics_to_grid=tmp_)
        assert res[0]["names_chronics_to_backend"] == tmp_
        
        # test correctly set from config
        with tempfile.TemporaryDirectory() as tmp:
            shutil.copy(
                f'{TEST_DEV_ENVS["l2rpn_case14_sandbox"]}/grid.json',
                os.path.join(tmp, "grid.json")
                )
            os.mkdir(os.path.join(tmp, "chronics"))
            
            with open(os.path.join(tmp, "config.py"), "w") as f:
                f.write("\n")
                f.write("config ={'names_chronics_to_grid': {'2': '1'} }\n")
            res = self.call_get_def_kwargs(tmp)  
            assert res[0]["names_chronics_to_backend"] == tmp_
        
    def test_grid_path(self):
        tmp_ = os.path.join(TEST_DEV_ENVS['l2rpn_icaps_2021'], "grid.json")
        res = self.call_get_def_kwargs(grid_path=tmp_)
        assert res[0]["init_grid_path"] == tmp_
        
        # test correctly set from config
        with tempfile.TemporaryDirectory() as tmp:
            shutil.copy(
                f'{TEST_DEV_ENVS["l2rpn_case14_sandbox"]}/grid.json',
                os.path.join(tmp, "grid.json")
                )
            os.mkdir(os.path.join(tmp, "chronics"))
            
            with open(os.path.join(tmp, "config.py"), "w") as f:
                f.write("\n")
                f.write(f"config = {{'grid_path': \"{tmp_}\" }}\n")
            res = self.call_get_def_kwargs(tmp)  
            assert res[0]["init_grid_path"] == tmp_
    
    def test_observation_class(self):
        # test correctly set from kwargs
        res = self.call_get_def_kwargs(observation_class=_MyObs)
        assert issubclass(res[0]["observationClass"], _MyObs)
        assert issubclass(_MyObs, res[0]["observationClass"])
        
        # test correctly set from config
        with tempfile.TemporaryDirectory() as tmp:
            shutil.copy(
                f'{TEST_DEV_ENVS["l2rpn_case14_sandbox"]}/grid.json',
                os.path.join(tmp, "grid.json")
                )
            os.mkdir(os.path.join(tmp, "chronics"))
            
            with open(os.path.join(tmp, "config.py"), "w") as f:
                f.write("from grid2op.tests.test_get_default_env_kwargs import _MyObs\n")
                f.write("config ={'observation_class': _MyObs}\n")
            res = self.call_get_def_kwargs(tmp)  
            assert hasattr(res[0]["observationClass"], "IS_MY_OBS")
            assert res[0]["observationClass"].IS_MY_OBS
        
    def test_param(self):
        # from make
        tmp_ = Parameters()
        tmp_.MAX_LINE_STATUS_CHANGED = 1234
        res = self.call_get_def_kwargs(param=tmp_)
        assert res[0]["parameters"] == tmp_
        
        # test fails if set from config
        with tempfile.TemporaryDirectory() as tmp:
            shutil.copy(
                f'{TEST_DEV_ENVS["l2rpn_case14_sandbox"]}/grid.json',
                os.path.join(tmp, "grid.json")
                )
            os.mkdir(os.path.join(tmp, "chronics"))
            
            with open(os.path.join(tmp, "config.py"), "w") as f:
                f.write("from grid2op.Parameters import Parameters\n")
                f.write("config = {'param': Parameters()}\n")
            # should be set with "parameters.json"
            # or with "difficulty_levels.json"
            with self.assertRaises(EnvError):
                res = self.call_get_def_kwargs(tmp)  
                
        # test correct way to set it from config
        with tempfile.TemporaryDirectory() as tmp:
            shutil.copy(
                f'{TEST_DEV_ENVS["l2rpn_case14_sandbox"]}/grid.json',
                os.path.join(tmp, "grid.json")
                )
            os.mkdir(os.path.join(tmp, "chronics"))
            with open(os.path.join(tmp, "parameters.json"), "w") as f:
                json.dump(obj=tmp_.to_dict(), fp=f)
            
            with open(os.path.join(tmp, "config.py"), "w") as f:
                f.write("\n")
                f.write("config ={}\n")
            res = self.call_get_def_kwargs(tmp)  
            assert res[0]["parameters"] == tmp_
        