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
from grid2op.Chronics.changeNothing import ChangeNothing
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
from grid2op.Rules.DefaultRules import DefaultRules
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


class _MyRules(DefaultRules):
    IS_MY_RULES = True


class _MyReward(BaseReward):
    IS_MY_REWARD = True


class _MyAction(BaseAction):
    IS_MY_ACTION = True


class _MyVoltageControler(ControlVoltageFromFile):
    IS_MY_VOLTAGE = True


class _MyChronicsClass(ChangeNothing):
    IS_MY_CHRONICS = True


class _MyOpponentSpace(OpponentSpace):
    IS_MY_OPP_SPACE = True


class _MyOpponent(BaseOpponent):
    IS_MY_OPP = True


class _MyOpponentBudget(BaseActionBudget):
    IS_MY_OPP_BUDGET = True


class _MyAttentionBudget(LinearAttentionBudget):
    IS_MY_ATTN = True
        
        
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
            _warn_layout_missing=False,
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

    def test_gamerules_class(self):
        # test correctly set from kwargs
        res = self.call_get_def_kwargs(gamerules_class=_MyRules)
        assert res[0]["legalActClass"] is _MyRules

        # test correctly set from config
        with tempfile.TemporaryDirectory() as tmp:
            shutil.copy(
                f'{TEST_DEV_ENVS["l2rpn_case14_sandbox"]}/grid.json',
                os.path.join(tmp, "grid.json")
            )
            os.mkdir(os.path.join(tmp, "chronics"))
            with open(os.path.join(tmp, "config.py"), "w") as f:
                f.write("from grid2op.tests.test_get_default_env_kwargs import _MyRules\n")
                f.write("config = {'gamerules_class': _MyRules}\n")
            res = self.call_get_def_kwargs(tmp)
            assert hasattr(res[0]["legalActClass"], "IS_MY_RULES")
            assert res[0]["legalActClass"].IS_MY_RULES

    def test_reward_class(self):
        # test correctly set from kwargs
        res = self.call_get_def_kwargs(reward_class=_MyReward)
        assert res[0]["rewardClass"] is _MyReward

        # test correctly set from config
        with tempfile.TemporaryDirectory() as tmp:
            shutil.copy(
                f'{TEST_DEV_ENVS["l2rpn_case14_sandbox"]}/grid.json',
                os.path.join(tmp, "grid.json")
            )
            os.mkdir(os.path.join(tmp, "chronics"))
            with open(os.path.join(tmp, "config.py"), "w") as f:
                f.write("from grid2op.tests.test_get_default_env_kwargs import _MyReward\n")
                f.write("config = {'reward_class': _MyReward}\n")
            res = self.call_get_def_kwargs(tmp)
            assert hasattr(res[0]["rewardClass"], "IS_MY_REWARD")
            assert res[0]["rewardClass"].IS_MY_REWARD

    def test_action_class(self):
        # test correctly set from kwargs
        res = self.call_get_def_kwargs(action_class=_MyAction)
        assert res[0]["actionClass"] is _MyAction

        # test correctly set from config
        with tempfile.TemporaryDirectory() as tmp:
            shutil.copy(
                f'{TEST_DEV_ENVS["l2rpn_case14_sandbox"]}/grid.json',
                os.path.join(tmp, "grid.json")
            )
            os.mkdir(os.path.join(tmp, "chronics"))
            with open(os.path.join(tmp, "config.py"), "w") as f:
                f.write("from grid2op.tests.test_get_default_env_kwargs import _MyAction\n")
                f.write("config = {'action_class': _MyAction}\n")
            res = self.call_get_def_kwargs(tmp)
            assert hasattr(res[0]["actionClass"], "IS_MY_ACTION")
            assert res[0]["actionClass"].IS_MY_ACTION

    def test_voltagecontroler_class(self):
        # test correctly set from kwargs
        res = self.call_get_def_kwargs(voltagecontroler_class=_MyVoltageControler)
        assert res[0]["voltagecontrolerClass"] is _MyVoltageControler

        # test correctly set from config
        with tempfile.TemporaryDirectory() as tmp:
            shutil.copy(
                f'{TEST_DEV_ENVS["l2rpn_case14_sandbox"]}/grid.json',
                os.path.join(tmp, "grid.json")
            )
            os.mkdir(os.path.join(tmp, "chronics"))
            with open(os.path.join(tmp, "config.py"), "w") as f:
                f.write("from grid2op.tests.test_get_default_env_kwargs import _MyVoltageControler\n")
                f.write("config = {'voltagecontroler_class': _MyVoltageControler}\n")
            res = self.call_get_def_kwargs(tmp)
            assert hasattr(res[0]["voltagecontrolerClass"], "IS_MY_VOLTAGE")
            assert res[0]["voltagecontrolerClass"].IS_MY_VOLTAGE

    def test_chronics_class(self):
        # test correctly set from kwargs
        res = self.call_get_def_kwargs(chronics_class=_MyChronicsClass)
        assert res[3].chronicsClass is _MyChronicsClass

        # test correctly set from config
        with tempfile.TemporaryDirectory() as tmp:
            shutil.copy(
                f'{TEST_DEV_ENVS["l2rpn_case14_sandbox"]}/grid.json',
                os.path.join(tmp, "grid.json")
            )
            os.mkdir(os.path.join(tmp, "chronics"))
            with open(os.path.join(tmp, "config.py"), "w") as f:
                f.write("from grid2op.tests.test_get_default_env_kwargs import _MyChronicsClass\n")
                f.write("config = {'chronics_class': _MyChronicsClass}\n")
            res = self.call_get_def_kwargs(tmp)
            assert hasattr(res[3].chronicsClass, "IS_MY_CHRONICS")
            assert res[3].chronicsClass.IS_MY_CHRONICS

    def test_other_rewards(self):
        # test correctly set from kwargs
        tmp_ = {'my_reward': BaseReward}
        res = self.call_get_def_kwargs(other_rewards=tmp_)
        assert res[0]["other_rewards"] == tmp_

        # test correctly set from config
        with tempfile.TemporaryDirectory() as tmp:
            shutil.copy(
                f'{TEST_DEV_ENVS["l2rpn_case14_sandbox"]}/grid.json',
                os.path.join(tmp, "grid.json")
            )
            os.mkdir(os.path.join(tmp, "chronics"))
            with open(os.path.join(tmp, "config.py"), "w") as f:
                f.write("from grid2op.Reward.baseReward import BaseReward\n")
                f.write("config = {'other_rewards': {'my_reward': BaseReward}}\n")
            res = self.call_get_def_kwargs(tmp)
            assert 'my_reward' in res[0]["other_rewards"]
            assert res[0]["other_rewards"]['my_reward'] is BaseReward

    def test_opponent_space_type(self):
        # test correctly set from kwargs
        res = self.call_get_def_kwargs(opponent_space_type=_MyOpponentSpace)
        assert res[0]["opponent_space_type"] is _MyOpponentSpace

        # test correctly set from config
        with tempfile.TemporaryDirectory() as tmp:
            shutil.copy(
                f'{TEST_DEV_ENVS["l2rpn_case14_sandbox"]}/grid.json',
                os.path.join(tmp, "grid.json")
            )
            os.mkdir(os.path.join(tmp, "chronics"))
            with open(os.path.join(tmp, "config.py"), "w") as f:
                f.write("from grid2op.tests.test_get_default_env_kwargs import _MyOpponentSpace\n")
                f.write("config = {'opponent_space_type': _MyOpponentSpace}\n")
            res = self.call_get_def_kwargs(tmp)
            assert hasattr(res[0]["opponent_space_type"], "IS_MY_OPP_SPACE")
            assert res[0]["opponent_space_type"].IS_MY_OPP_SPACE

    def test_opponent_action_class(self):
        # test correctly set from kwargs
        res = self.call_get_def_kwargs(opponent_action_class=_MyAction)
        assert res[0]["opponent_action_class"] is _MyAction

        # test correctly set from config
        with tempfile.TemporaryDirectory() as tmp:
            shutil.copy(
                f'{TEST_DEV_ENVS["l2rpn_case14_sandbox"]}/grid.json',
                os.path.join(tmp, "grid.json")
            )
            os.mkdir(os.path.join(tmp, "chronics"))
            with open(os.path.join(tmp, "config.py"), "w") as f:
                f.write("from grid2op.tests.test_get_default_env_kwargs import _MyAction\n")
                f.write("config = {'opponent_action_class': _MyAction}\n")
            res = self.call_get_def_kwargs(tmp)
            assert hasattr(res[0]["opponent_action_class"], "IS_MY_ACTION")
            assert res[0]["opponent_action_class"].IS_MY_ACTION

    def test_opponent_class(self):
        # test correctly set from kwargs
        res = self.call_get_def_kwargs(opponent_class=_MyOpponent)
        assert res[0]["opponent_class"] is _MyOpponent

        # test correctly set from config
        with tempfile.TemporaryDirectory() as tmp:
            shutil.copy(
                f'{TEST_DEV_ENVS["l2rpn_case14_sandbox"]}/grid.json',
                os.path.join(tmp, "grid.json")
            )
            os.mkdir(os.path.join(tmp, "chronics"))
            with open(os.path.join(tmp, "config.py"), "w") as f:
                f.write("from grid2op.tests.test_get_default_env_kwargs import _MyOpponent\n")
                f.write("config = {'opponent_class': _MyOpponent}\n")
            res = self.call_get_def_kwargs(tmp)
            assert hasattr(res[0]["opponent_class"], "IS_MY_OPP")
            assert res[0]["opponent_class"].IS_MY_OPP

    def test_opponent_budget_class(self):
        # test correctly set from kwargs
        res = self.call_get_def_kwargs(opponent_budget_class=_MyOpponentBudget)
        assert res[0]["opponent_budget_class"] is _MyOpponentBudget

        # test correctly set from config
        with tempfile.TemporaryDirectory() as tmp:
            shutil.copy(
                f'{TEST_DEV_ENVS["l2rpn_case14_sandbox"]}/grid.json',
                os.path.join(tmp, "grid.json")
            )
            os.mkdir(os.path.join(tmp, "chronics"))
            with open(os.path.join(tmp, "config.py"), "w") as f:
                f.write("from grid2op.tests.test_get_default_env_kwargs import _MyOpponentBudget\n")
                f.write("config = {'opponent_budget_class': _MyOpponentBudget}\n")
            res = self.call_get_def_kwargs(tmp)
            assert hasattr(res[0]["opponent_budget_class"], "IS_MY_OPP_BUDGET")
            assert res[0]["opponent_budget_class"].IS_MY_OPP_BUDGET

    def test_opponent_init_budget(self):
        # test correctly set from kwargs
        res = self.call_get_def_kwargs(opponent_init_budget=123.4)
        assert res[0]["opponent_init_budget"] == 123.4

        # test correctly set from config
        with tempfile.TemporaryDirectory() as tmp:
            shutil.copy(
                f'{TEST_DEV_ENVS["l2rpn_case14_sandbox"]}/grid.json',
                os.path.join(tmp, "grid.json")
            )
            os.mkdir(os.path.join(tmp, "chronics"))
            with open(os.path.join(tmp, "config.py"), "w") as f:
                f.write("config = {'opponent_init_budget': 123.4}\n")
            res = self.call_get_def_kwargs(tmp)
            assert res[0]["opponent_init_budget"] == 123.4

    def test_opponent_budget_per_ts(self):
        # test correctly set from kwargs
        res = self.call_get_def_kwargs(opponent_budget_per_ts=0.5)
        assert res[0]["opponent_budget_per_ts"] == 0.5

        # test correctly set from config
        with tempfile.TemporaryDirectory() as tmp:
            shutil.copy(
                f'{TEST_DEV_ENVS["l2rpn_case14_sandbox"]}/grid.json',
                os.path.join(tmp, "grid.json")
            )
            os.mkdir(os.path.join(tmp, "chronics"))
            with open(os.path.join(tmp, "config.py"), "w") as f:
                f.write("config = {'opponent_budget_per_ts': 0.5}\n")
            res = self.call_get_def_kwargs(tmp)
            assert res[0]["opponent_budget_per_ts"] == 0.5

    def test_opponent_attack_duration(self):
        # test correctly set from kwargs
        res = self.call_get_def_kwargs(opponent_attack_duration=5)
        assert res[0]["opponent_attack_duration"] == 5

        # test correctly set from config
        with tempfile.TemporaryDirectory() as tmp:
            shutil.copy(
                f'{TEST_DEV_ENVS["l2rpn_case14_sandbox"]}/grid.json',
                os.path.join(tmp, "grid.json")
            )
            os.mkdir(os.path.join(tmp, "chronics"))
            with open(os.path.join(tmp, "config.py"), "w") as f:
                f.write("config = {'opponent_attack_duration': 5}\n")
            res = self.call_get_def_kwargs(tmp)
            assert res[0]["opponent_attack_duration"] == 5

    def test_opponent_attack_cooldown(self):
        # test correctly set from kwargs
        res = self.call_get_def_kwargs(opponent_attack_cooldown=10)
        assert res[0]["opponent_attack_cooldown"] == 10

        # test correctly set from config
        with tempfile.TemporaryDirectory() as tmp:
            shutil.copy(
                f'{TEST_DEV_ENVS["l2rpn_case14_sandbox"]}/grid.json',
                os.path.join(tmp, "grid.json")
            )
            os.mkdir(os.path.join(tmp, "chronics"))
            with open(os.path.join(tmp, "config.py"), "w") as f:
                f.write("config = {'opponent_attack_cooldown': 10}\n")
            res = self.call_get_def_kwargs(tmp)
            assert res[0]["opponent_attack_cooldown"] == 10

    def test_kwargs_opponent(self):
        # test correctly set from kwargs
        tmp_ = {'key': 'value'}
        res = self.call_get_def_kwargs(kwargs_opponent=tmp_)
        assert res[0]["kwargs_opponent"] == tmp_

        # test correctly set from config
        with tempfile.TemporaryDirectory() as tmp:
            shutil.copy(
                f'{TEST_DEV_ENVS["l2rpn_case14_sandbox"]}/grid.json',
                os.path.join(tmp, "grid.json")
            )
            os.mkdir(os.path.join(tmp, "chronics"))
            with open(os.path.join(tmp, "config.py"), "w") as f:
                f.write("config = {'kwargs_opponent': {'key': 'value'}}\n")
            res = self.call_get_def_kwargs(tmp)
            assert res[0]["kwargs_opponent"] == {'key': 'value'}

    def test_has_attention_budget(self):
        # test correctly set from kwargs
        res = self.call_get_def_kwargs(has_attention_budget=True)
        assert res[0]["has_attention_budget"] is True

        # test correctly set from config
        with tempfile.TemporaryDirectory() as tmp:
            shutil.copy(
                f'{TEST_DEV_ENVS["l2rpn_case14_sandbox"]}/grid.json',
                os.path.join(tmp, "grid.json")
            )
            os.mkdir(os.path.join(tmp, "chronics"))
            with open(os.path.join(tmp, "config.py"), "w") as f:
                f.write("config = {'has_attention_budget': True}\n")
            res = self.call_get_def_kwargs(tmp)
            assert res[0]["has_attention_budget"] is True

    def test_attention_budget_class(self):
        # test correctly set from kwargs
        res = self.call_get_def_kwargs(attention_budget_class=_MyAttentionBudget)
        assert res[0]["attention_budget_cls"] is _MyAttentionBudget

        # test correctly set from config
        with tempfile.TemporaryDirectory() as tmp:
            shutil.copy(
                f'{TEST_DEV_ENVS["l2rpn_case14_sandbox"]}/grid.json',
                os.path.join(tmp, "grid.json")
            )
            os.mkdir(os.path.join(tmp, "chronics"))
            with open(os.path.join(tmp, "config.py"), "w") as f:
                f.write("from grid2op.tests.test_get_default_env_kwargs import _MyAttentionBudget\n")
                f.write("config = {'attention_budget_class': _MyAttentionBudget}\n")
            res = self.call_get_def_kwargs(tmp)
            assert hasattr(res[0]["attention_budget_cls"], "IS_MY_ATTN")
            assert res[0]["attention_budget_cls"].IS_MY_ATTN

    def test_kwargs_attention_budget(self):
        # test correctly set from kwargs
        tmp_ = {'key': 'value'}
        res = self.call_get_def_kwargs(kwargs_attention_budget=tmp_)
        assert res[0]["kwargs_attention_budget"] == tmp_

        # test correctly set from config
        with tempfile.TemporaryDirectory() as tmp:
            shutil.copy(
                f'{TEST_DEV_ENVS["l2rpn_case14_sandbox"]}/grid.json',
                os.path.join(tmp, "grid.json")
            )
            os.mkdir(os.path.join(tmp, "chronics"))
            with open(os.path.join(tmp, "config.py"), "w") as f:
                f.write("config = {'kwargs_attention_budget': {'key': 'value'}}\n")
            res = self.call_get_def_kwargs(tmp)
            assert res[0]["kwargs_attention_budget"] == {'key': 'value'}

    def test_kwargs_observation(self):
        # test correctly set from kwargs
        tmp_ = {'key': 'value'}
        res = self.call_get_def_kwargs(kwargs_observation=tmp_)
        assert res[0]["kwargs_observation"] == tmp_

        # kwargs_observation is not read from config, default is {}
        with tempfile.TemporaryDirectory() as tmp:
            shutil.copy(
                f'{TEST_DEV_ENVS["l2rpn_case14_sandbox"]}/grid.json',
                os.path.join(tmp, "grid.json")
            )
            os.mkdir(os.path.join(tmp, "chronics"))
            with open(os.path.join(tmp, "config.py"), "w") as f:
                f.write("config = {}\n")
            res = self.call_get_def_kwargs(tmp)
            assert res[0]["kwargs_observation"] == {}

    def test_observation_backend_class(self):
        # test correctly set from kwargs
        res = self.call_get_def_kwargs(observation_backend_class=_MyBack)
        assert res[0]["observation_bk_class"] is _MyBack

        # test correctly set from config
        with tempfile.TemporaryDirectory() as tmp:
            shutil.copy(
                f'{TEST_DEV_ENVS["l2rpn_case14_sandbox"]}/grid.json',
                os.path.join(tmp, "grid.json")
            )
            os.mkdir(os.path.join(tmp, "chronics"))
            with open(os.path.join(tmp, "config.py"), "w") as f:
                f.write("from grid2op.tests.test_get_default_env_kwargs import _MyBack\n")
                f.write("config = {'observation_backend_class': _MyBack}\n")
            res = self.call_get_def_kwargs(tmp)
            assert hasattr(res[0]["observation_bk_class"], "IS_MY_BACKEND")
            assert res[0]["observation_bk_class"].IS_MY_BACKEND

    def test_observation_backend_kwargs(self):
        # test correctly set from kwargs
        tmp_ = {'key': 'value'}
        res = self.call_get_def_kwargs(observation_backend_kwargs=tmp_)
        assert res[0]["observation_bk_kwargs"] == tmp_

        # test default is None when not provided
        res = self.call_get_def_kwargs()
        assert res[0]["observation_bk_kwargs"] is None

        # test correctly set from config
        with tempfile.TemporaryDirectory() as tmp:
            shutil.copy(
                f'{TEST_DEV_ENVS["l2rpn_case14_sandbox"]}/grid.json',
                os.path.join(tmp, "grid.json")
            )
            os.mkdir(os.path.join(tmp, "chronics"))
            with open(os.path.join(tmp, "config.py"), "w") as f:
                f.write("config = {'observation_backend_kwargs': {'key': 'value'}}\n")
            res = self.call_get_def_kwargs(tmp)
            assert res[0]["observation_bk_kwargs"] == {'key': 'value'}
        