# Copyright (c) 2019-2023, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import warnings
import numpy as np
import unittest
import os
import tempfile
from grid2op.tests.helper_path_test import *

from grid2op import make
from grid2op.Reward import AlertReward
from grid2op.Parameters import Parameters
from grid2op.Exceptions import Grid2OpException
from grid2op.Runner import Runner  # TODO
from grid2op.Opponent import BaseOpponent, GeometricOpponent
from grid2op.Action import PlayableAction

ALL_ATTACKABLE_LINES= [
            "62_58_180",
            "62_63_160",
            "48_50_136",
            "48_53_141",
            "41_48_131",
            "39_41_121",
            "43_44_125",
            "44_45_126",
            "34_35_110",
            "54_58_154",
        ] 

ATTACKED_LINE = "48_50_136"


def _get_steps_attack(kwargs_opponent, multi=False):
    """computes the steps for which there will be attacks"""
    ts_attack = np.array(kwargs_opponent["steps_attack"])
    res = []
    for i, ts in enumerate(ts_attack):
        if not multi:
            res.append(ts + np.arange(kwargs_opponent["duration"]))
        else:
            res.append(ts + np.arange(kwargs_opponent["duration"][i]))
    return np.unique(np.concatenate(res).flatten())

    
class TestOpponent(BaseOpponent): 
    """An opponent that can select the line attack, the time and duration of the attack."""
    
    def __init__(self, action_space):
        super().__init__(action_space)
        self.custom_attack = None
        self.duration = None
        self.steps_attack = None

    def init(self, partial_env,  lines_attacked=[ATTACKED_LINE], duration=10, steps_attack=[0,1]):
        attacked_line = lines_attacked[0]
        self.custom_attack = self.action_space({"set_line_status" : [(l, -1) for l in lines_attacked]})
        self.duration = duration
        self.steps_attack = steps_attack
        self.env = partial_env

    def attack(self, observation, agent_action, env_action, budget, previous_fails): 
        if observation is None:
            return None, None
        current_step = self.env.nb_time_step
        if current_step not in self.steps_attack: 
            return None, None
        
        return self.custom_attack, self.duration

class TestOpponentMultiLines(BaseOpponent): 
    """An opponent that can select the line attack, the time and duration of the attack."""
    
    def __init__(self, action_space):
        super().__init__(action_space)
        self.custom_attack = None
        self.duration = None
        self.steps_attack = None

    def init(self, partial_env,  lines_attacked=[ATTACKED_LINE], duration=[10,10], steps_attack=[0,1]):
        attacked_line = lines_attacked[0]
        self.custom_attack = [ self.action_space({"set_line_status" : [(l, -1)]}) for l in lines_attacked]
        self.duration = duration
        self.steps_attack = steps_attack
        self.env = partial_env
        
    def attack(self, observation, agent_action, env_action, budget, previous_fails): 
        if observation is None:
            return None, None

        current_step = self.env.nb_time_step
        if current_step not in self.steps_attack: 
            return None, None
        
        index = self.steps_attack.index(current_step)

        return self.custom_attack[index], self.duration[index]

# Test alert blackout / tets alert no blackout
class TestAlertNoBlackout(unittest.TestCase):
    """test the basic bahavior of the assistant alert feature when no attack occur """

    def setUp(self) -> None:
        self.env_nm = os.path.join(
            PATH_DATA_TEST, "l2rpn_idf_2023_with_alert"
        )

    def test_init_default_param(self) -> None : 
        with make(self.env_nm, test=True, difficulty="1") as env:
            assert isinstance(env.parameters.ALERT_TIME_WINDOW, np.int32)
            assert env._attention_budget is None  # no attention budget for the alert
            assert env._opponent_class == GeometricOpponent
            assert env.parameters.ALERT_TIME_WINDOW > 0

            param = env.parameters
            param.init_from_dict({"ALERT_TIME_WINDOW": -1})
            
            negative_value_invalid = False
            try: 
                env.change_parameters(param)
                env.reset()
            except : 
                negative_value_invalid = True 

            assert negative_value_invalid

            # test observations for this env also
            true_alertable_lines = ALL_ATTACKABLE_LINES
            
            assert isinstance(env.alertable_line_names, list)
            assert sorted(env.alertable_line_names) == sorted(true_alertable_lines)
            assert env.dim_alerts == len(true_alertable_lines)


    def test_init_observation(self) -> None :    
        true_alertable_lines = [ATTACKED_LINE]
        kwargs_opponent = dict(lines_attacked=[ATTACKED_LINE], 
                                duration=10, 
                                steps_attack=[0,10])
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with make(self.env_nm,
                      test=True,
                      difficulty="1", 
                      opponent_attack_cooldown=0, 
                      opponent_attack_duration=99999, 
                      opponent_budget_per_ts=1000, 
                      opponent_init_budget=10000., 
                      opponent_action_class=PlayableAction, 
                      opponent_class=TestOpponent, 
                      kwargs_opponent=kwargs_opponent, 
                      reward_class=AlertReward(reward_end_episode_bonus=42),
                      _add_to_name="_tio") as env:
                assert isinstance(env.alertable_line_names, list)
                assert sorted(env.alertable_line_names) == sorted(true_alertable_lines)
                assert env.dim_alerts == len(true_alertable_lines)


    def test_raise_alert_action(self) -> None :
        kwargs_opponent = dict(lines_attacked=[ATTACKED_LINE], 
                               duration=10, 
                               steps_attack=[0,10])
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with make(self.env_nm,
                      test=True,
                      difficulty="1", 
                      opponent_attack_cooldown=0, 
                      opponent_attack_duration=99999, 
                      opponent_budget_per_ts=1000, 
                      opponent_init_budget=10000., 
                      opponent_action_class=PlayableAction, 
                      opponent_class=TestOpponent, 
                      kwargs_opponent=kwargs_opponent, 
                      reward_class=AlertReward(reward_end_episode_bonus=42),
                      _add_to_name="_traa") as env:
                
                for attackable_line_id in range(env.dim_alerts):
                    # raise alert on line number line_id
                    act = env.action_space()
                    act.raise_alert = [attackable_line_id]

                    act_2 = env.action_space({"raise_alert": [attackable_line_id]})
                    
                    assert act == act_2, f"error for line {attackable_line_id}"

    def test_assistant_reward_value_no_blackout_no_attack_no_alert(self) -> None : 
        """ When no blackout and no attack occur, and no alert is raised we expect a reward of 0
            until the end of the episode where we have a bonus (here artificially 42)

        Raises:
            Grid2OpException: raise an exception if an attack occur
        """
        with make(
            self.env_nm,
            test=True,
            difficulty="1",
            reward_class=AlertReward(reward_end_episode_bonus=42)
        ) as env:
            env.seed(0)
            env.reset()

            done = False
            for i in range(env.max_episode_duration()):
                obs, reward, done, info = env.step(env.action_space())
                if info["opponent_attack_line"] is None : 
                    if i == env.max_episode_duration()-1: 
                        assert reward == 42
                    else : 
                        assert reward == 0
                else : 
                    raise Grid2OpException('No attack expected')
            
            assert done
    
    def test_assistant_reward_value_no_blackout_no_attack_alert(self) -> None : 
        """ When an alert is raised while no attack / nor blackout occur, we expect a reward of 0
            until the end of the episode where we have a bonus (here artificially 42)

        Raises:
            Grid2OpException: raise an exception if an attack occur
        """
        with make(
            self.env_nm,
            test=True,
            difficulty="1",
            reward_class=AlertReward(reward_end_episode_bonus=42)
        ) as env:
            env.seed(0)
            env.reset()

            done = False
            attackable_line_id=0
            for i in range(env.max_episode_duration()):
                act = env.action_space()
                if i == 1 :
                    act = env.action_space({"raise_alert": [attackable_line_id]})
                obs, reward, done, info = env.step(act)

                if info["opponent_attack_line"] is None : 
                    if i == env.max_episode_duration()-1: 
                        assert reward == 42
                    else : 
                        assert reward == 0
                else : 
                    raise Grid2OpException('No attack expected')
            
            assert done


# If attack 
    def test_assistant_reward_value_no_blackout_attack_no_alert(self) -> None :
        """ When we don't raise an alert for an attack but no blackout occur, we expect a reward of 1
            at step 3 (here with a window size of 2)
            otherwise 0 at other time steps
            until the end of the episode where we have a bonus (here artificially 42)

        """
        kwargs_opponent = dict(lines_attacked=[ATTACKED_LINE], 
                                   duration=3, 
                                   steps_attack=[1])
        with make(self.env_nm,
                  test=True,
                  difficulty="1", 
                  opponent_attack_cooldown=0, 
                  opponent_attack_duration=99999, 
                  opponent_budget_per_ts=1000, 
                  opponent_init_budget=10000., 
                  opponent_action_class=PlayableAction, 
                  opponent_class=TestOpponent, 
                  kwargs_opponent=kwargs_opponent,
                  reward_class=AlertReward(reward_end_episode_bonus=42),
                  _add_to_name="_tarvnbana"
            ) as env : 
            env.seed(0)
            env.reset()
            step = 0
            for i in range(env.max_episode_duration()):
                act = env.action_space()
                obs, reward, done, info = env.step(act)
                step += 1
                if step in _get_steps_attack(kwargs_opponent): 
                    assert info["opponent_attack_line"] is not None, f"no attack is detected at step {step}"
                else:
                    assert info["opponent_attack_line"]  is None, f"an attack is detected at step {step}"
                    
                if step == 3 : 
                    assert reward == 1
                elif step == env.max_episode_duration(): 
                        assert reward == 42
                else : 
                    assert reward == 0

    def test_assistant_reward_value_no_blackout_attack_alert(self) -> None :
        """When an alert occur at step 2, we raise an alert is at step 1 
            We expect a reward -1 at step 3 (with a window size of 2)
        """
        kwargs_opponent = dict(lines_attacked=[ATTACKED_LINE], 
                               duration=3, 
                               steps_attack=[2])
        with make(self.env_nm,
                  test=True,
                  difficulty="1", 
                  opponent_attack_cooldown=0, 
                  opponent_attack_duration=99999, 
                  opponent_budget_per_ts=1000, 
                  opponent_init_budget=10000., 
                  opponent_action_class=PlayableAction, 
                  opponent_class=TestOpponent, 
                  kwargs_opponent=kwargs_opponent,
                  _add_to_name="_tarvnba"
            ) as env : 
            env.seed(0)
            env.reset()
            step = 0
            for i in range(env.max_episode_duration()):
                attackable_line_id = 0
                act = env.action_space()
                if i == 1 :
                    act = env.action_space({"raise_alert": [attackable_line_id]})
                obs, reward, done, info = env.step(act)
                step += 1
                if step in _get_steps_attack(kwargs_opponent): 
                    assert info["opponent_attack_line"] is not None, f"no attack is detected at step {step}"
                else:
                    assert info["opponent_attack_line"]  is None, f"an attack is detected at step {step}"
                    
                if step == 4 : 
                    assert reward == -1
                elif step == env.max_episode_duration(): 
                    assert reward == 1
                else : 
                    assert reward == 0

    def test_assistant_reward_value_no_blackout_attack_alert_too_late(self) -> None :
        """ When we raise an alert too late for an attack but no blackout occur, we expect a reward of 1
            at step 3 (here with a window size of 2)
            otherwise 0 at other time steps
            until the end of the episode where we have a bonus (here artificially 42)

        """
        kwargs_opponent = dict(lines_attacked=[ATTACKED_LINE], 
                                   duration=3, 
                                   steps_attack=[2])
        with make(self.env_nm,
                  test=True,
                  difficulty="1", 
                  opponent_attack_cooldown=0, 
                  opponent_attack_duration=99999, 
                  opponent_budget_per_ts=1000, 
                  opponent_init_budget=10000., 
                  opponent_action_class=PlayableAction, 
                  opponent_class=TestOpponent, 
                  kwargs_opponent=kwargs_opponent,
                  _add_to_name="_tarvnbaatl"
            ) as env : 
            env.seed(0)
            env.reset()
            step = 0
            for i in range(env.max_episode_duration()):
                attackable_line_id = 0
                act = env.action_space()
                if i == 2 :
                    act = env.action_space({"raise_alert": [attackable_line_id]})
                obs, reward, done, info = env.step(act)
                step += 1
                if step in _get_steps_attack(kwargs_opponent): 
                    assert info["opponent_attack_line"] is not None, f"no attack is detected at step {step}"
                else:
                    assert info["opponent_attack_line"]  is None, f"an attack is detected at step {step}"
                    
                if step == 4 : 
                    assert reward == 1
                elif step == env.max_episode_duration(): 
                    assert reward == 1
                else : 
                    assert reward == 0

    def test_assistant_reward_value_no_blackout_attack_alert_too_early(self)-> None :
        """ When we raise an alert too early for an attack but no blackout occur, we expect a reward of 1
            at step 3 (here with a window size of 2)
            otherwise 0 at other time steps
            until the end of the episode where we have a bonus (here artificially 42)

        """
        kwargs_opponent = dict(lines_attacked=[ATTACKED_LINE], 
                                   duration=3, 
                                   steps_attack=[2])
        with make(self.env_nm,
                  test=True,
                  difficulty="1", 
                  opponent_attack_cooldown=0, 
                  opponent_attack_duration=99999, 
                  opponent_budget_per_ts=1000, 
                  opponent_init_budget=10000., 
                  opponent_action_class=PlayableAction, 
                  opponent_class=TestOpponent, 
                  kwargs_opponent=kwargs_opponent,
                  _add_to_name="_tarvnbaate"
            ) as env : 
            env.seed(0)
            env.reset()
            step = 0
            for i in range(env.max_episode_duration()):
                attackable_line_id = 0
                act = env.action_space()
                if i == 0 :
                    # An alert is raised at step 0
                    act = env.action_space({"raise_alert": [attackable_line_id]})
                obs, reward, done, info = env.step(act)
                step += 1
                if step in _get_steps_attack(kwargs_opponent): 
                    assert info["opponent_attack_line"] is not None, f"no attack is detected at step {step}"
                else:
                    assert info["opponent_attack_line"]  is None, f"an attack is detected at step {step}"
                    
                if step == 4: 
                    assert reward == 1
                elif step == env.max_episode_duration(): 
                    assert reward == 1
                else : 
                    assert reward == 0

    # 2 ligne attaquées 
    def test_assistant_reward_value_no_blackout_2_attack_same_time_no_alert(self) -> None :
        """ When we don't raise an alert for 2 attacks at the same time 
            but no blackout occur, we expect a reward of 1
            at step 3 (here with a window size of 2)
            otherwise 0 at other time steps
            until the end of the episode where we have a bonus (here artificially 42)
        """

        kwargs_opponent = dict(lines_attacked=[ATTACKED_LINE]+['48_53_141'], 
                                   duration=3, 
                                   steps_attack=[1])
        with make(self.env_nm,
                  test=True,
                  difficulty="1", 
                  opponent_attack_cooldown=0, 
                  opponent_attack_duration=99999, 
                  opponent_budget_per_ts=1000, 
                  opponent_init_budget=10000., 
                  opponent_action_class=PlayableAction, 
                  opponent_class=TestOpponent, 
                  kwargs_opponent=kwargs_opponent,
                  reward_class=AlertReward(reward_end_episode_bonus=42),
                  _add_to_name="_tarvnb2astna"
            ) as env : 
            env.seed(0)
            env.reset()
            step = 0
            for i in range(env.max_episode_duration()):
                act = env.action_space()
                obs, reward, done, info = env.step(act)
                step += 1
                if step in _get_steps_attack(kwargs_opponent): 
                    assert info["opponent_attack_line"] is not None, f"no attack is detected at step {step}"
                else:
                    assert info["opponent_attack_line"]  is None, f"an attack is detected at step {step}"
                    
                if step == 3 : 
                    assert reward == 1
                elif step == env.max_episode_duration(): 
                        assert reward == 42
                else : 
                    assert reward == 0
    
    def test_assistant_reward_value_no_blackout_2_attack_same_time_1_alert(self) -> None :
        """ When we raise only 1 alert for 2 attacks at the same time 
            but no blackout occur, we expect a reward of 0
            at step 3 (here with a window size of 2)
            otherwise 0 at other time steps
            until the end of the episode where we have a bonus (here artificially 42)
        """
        kwargs_opponent = dict(lines_attacked=[ATTACKED_LINE]+['48_53_141'], 
                               duration=3, 
                               steps_attack=[2])
        with make(self.env_nm,
                  test=True,
                  difficulty="1", 
                  opponent_attack_cooldown=0, 
                  opponent_attack_duration=99999, 
                  opponent_budget_per_ts=1000, 
                  opponent_init_budget=10000., 
                  opponent_action_class=PlayableAction, 
                  opponent_class=TestOpponent, 
                  kwargs_opponent=kwargs_opponent,
                  _add_to_name="_tarvnb2ast1a"
            ) as env : 
            env.seed(0)
            env.reset()
            step = 0
            for i in range(env.max_episode_duration()):
                attackable_line_id = 0
                act = env.action_space()
                if i == 1 :
                    act = env.action_space({"raise_alert": [attackable_line_id]})
                obs, reward, done, info = env.step(act)
                step += 1
                if step in _get_steps_attack(kwargs_opponent): 
                    assert info["opponent_attack_line"] is not None, f"no attack is detected at step {step}"
                else:
                    assert info["opponent_attack_line"]  is None, f"an attack is detected at step {step}"
                    
                if step == 4 : 
                    assert reward == 0
                elif step == env.max_episode_duration(): 
                    assert reward == 1
                else : 
                    assert reward == 0

    def test_assistant_reward_value_no_blackout_2_attack_same_time_2_alert(self) -> None :
        """ When we raise 2 alerts for 2 attacks at the same time 
            but no blackout occur, we expect a reward of -1
            at step 3 (here with a window size of 2)
            otherwise 0 at other time steps
            until the end of the episode where we have a bonus (here artificially 42)
        """
        kwargs_opponent = dict(lines_attacked=[ATTACKED_LINE]+['48_53_141'], 
                                   duration=3, 
                                   steps_attack=[2])
        with make(self.env_nm,
                  test=True,
                  difficulty="1", 
                  opponent_attack_cooldown=0, 
                  opponent_attack_duration=99999, 
                  opponent_budget_per_ts=1000, 
                  opponent_init_budget=10000., 
                  opponent_action_class=PlayableAction, 
                  opponent_class=TestOpponent, 
                  kwargs_opponent=kwargs_opponent,
                  _add_to_name="_tarvnb2ast2a"
            ) as env : 
            env.seed(0)
            env.reset()
            step = 0
            for i in range(env.max_episode_duration()):
                attackable_line_ids = [0, 1]
                act = env.action_space()
                if i == 1 :
                    act = env.action_space({"raise_alert": attackable_line_ids})
                obs, reward, done, info = env.step(act)
                step += 1
                if step in _get_steps_attack(kwargs_opponent): 
                    assert info["opponent_attack_line"] is not None, f"no attack is detected at step {step}"
                else:
                    assert info["opponent_attack_line"]  is None, f"an attack is detected at step {step}"
                    
                if step == 4 : 
                    assert reward == -1
                elif step == env.max_episode_duration(): 
                    assert reward == 1
                else : 
                    assert reward == 0


    def test_assistant_reward_value_no_blackout_2_attack_diff_time_no_alert(self) -> None :
        """ When we don't raise an alert for 2 attacks at two times  
            but no blackout occur, we expect a reward of 1
            at step 3 (here with a window size of 2)
            otherwise 0 at other time steps
            until the end of the episode where we have a bonus (here artificially 42)
        """

        kwargs_opponent = dict(lines_attacked=[ATTACKED_LINE]+['48_53_141'], 
                               duration=[1, 1], 
                               steps_attack=[1, 2])
        with make(self.env_nm,
                  test=True,
                  difficulty="1", 
                  opponent_attack_cooldown=0, 
                  opponent_attack_duration=99999, 
                  opponent_budget_per_ts=1000, 
                  opponent_init_budget=10000., 
                  opponent_action_class=PlayableAction, 
                  opponent_class=TestOpponentMultiLines, 
                  kwargs_opponent=kwargs_opponent,
                  reward_class=AlertReward(reward_end_episode_bonus=42),
                  _add_to_name="_tarvnb2dtna"
            ) as env : 
            env.seed(0)
            env.reset()
            step = 0
            for i in range(env.max_episode_duration()):
                act = env.action_space()
                obs, reward, done, info = env.step(act)
                step += 1
                
                if step in _get_steps_attack(kwargs_opponent, multi=True) : 
                    assert info["opponent_attack_line"] is not None, f"no attack is detected at step {step}"
                else:
                    assert info["opponent_attack_line"]  is None, f"an attack is detected at step {step}"
                    
                if step == 3 : 
                    assert reward == 1
                elif step == 4 : 
                    assert reward == 1
                elif step == env.max_episode_duration(): 
                        assert reward == 42
                else : 
                    assert reward == 0
        
    def test_assistant_reward_value_no_blackout_2_attack_diff_time_2_alert(self) -> None :
        """ When we raise 2 alert for 2 attacks at two times  
            but no blackout occur, we expect a reward of -1
            at step 3 (here with a window size of 2) and step 4
            otherwise 0 at other time steps
            until the end of the episode where we have a bonus (here artificially 42)
        """

        kwargs_opponent = dict(lines_attacked=[ATTACKED_LINE]+['48_53_141'], 
                               duration=[1,1], 
                               steps_attack=[2, 3])
        with make(self.env_nm,
                  test=True,
                  difficulty="1", 
                  opponent_attack_cooldown=0, 
                  opponent_attack_duration=99999, 
                  opponent_budget_per_ts=1000, 
                  opponent_init_budget=10000., 
                  opponent_action_class=PlayableAction, 
                  opponent_class=TestOpponentMultiLines, 
                  kwargs_opponent=kwargs_opponent,
                  reward_class=AlertReward(reward_end_episode_bonus=42),
                  _add_to_name="_tarvnb2dt2a"
            ) as env : 
            env.seed(0)
            env.reset()
            step = 0
            for i in range(env.max_episode_duration()):
                act = env.action_space()
                if i == 1 :
                    act = env.action_space({"raise_alert": [0]})
                elif i == 2 : 
                    act = env.action_space({"raise_alert": [1]})
                obs, reward, done, info = env.step(act)
                step += 1
                if step in _get_steps_attack(kwargs_opponent, multi=True): 
                    assert info["opponent_attack_line"] is not None, f"no attack is detected at step {step}"
                else:
                    assert info["opponent_attack_line"]  is None, f"an attack is detected at step {step}"
                    
                if step == 4: 
                    assert reward == -1, f"error for step {step}: {reward} instead of -1."
                elif step == 5: 
                    assert reward == -1
                elif step == env.max_episode_duration(): 
                        assert reward == 42, f"error for step {step}: {reward} instead of -1."
                else : 
                    assert reward == 0, f"error for step {step}: {reward} instead of 0."

    def test_assistant_reward_value_no_blackout_2_attack_diff_time_alert_first_attack(self) -> None :
        """ When we raise 1 alert on the first attack while we have 2 attacks at two times  
            but no blackout occur, we expect a reward of -1
            at step 3 (here with a window size of 2) and 1 step 4
            otherwise 0 at other time steps
            until the end of the episode where we have a bonus (here artificially 42)
        """

        kwargs_opponent = dict(lines_attacked=[ATTACKED_LINE]+['48_53_141'], 
                                   duration=[1,1], 
                                   steps_attack=[2, 3])
        with make(self.env_nm,
                  test=True,
                  difficulty="1", 
                  opponent_attack_cooldown=0, 
                  opponent_attack_duration=99999, 
                  opponent_budget_per_ts=1000, 
                  opponent_init_budget=10000., 
                  opponent_action_class=PlayableAction, 
                  opponent_class=TestOpponentMultiLines, 
                  kwargs_opponent=kwargs_opponent,
                  reward_class=AlertReward(reward_end_episode_bonus=42),
                  _add_to_name="_tarvnb2dtafa"
            ) as env : 
            env.seed(0)
            env.reset()
            step = 0
            for i in range(env.max_episode_duration()):
                act = env.action_space()
                if i == 1 :
                    act = env.action_space({"raise_alert": [0]})
                obs, reward, done, info = env.step(act)
                step += 1
                if step in _get_steps_attack(kwargs_opponent, multi=True): 
                    assert info["opponent_attack_line"] is not None, f"no attack is detected at step {step}"
                else:
                    assert info["opponent_attack_line"]  is None, f"an attack is detected at step {step}"
                    
                if step == 4 : 
                    assert reward == -1, f"error for step {step}: {reward} vs -1"
                elif step == 5 : 
                    assert reward == 1, f"error for step {step}: {reward} vs 1"
                elif step == env.max_episode_duration(): 
                        assert reward == 42, f"error for step {step}: {reward} vs 42"
                else : 
                    assert reward == 0, f"error for step {step}: {reward} vs 0"


    def test_assistant_reward_value_no_blackout_2_attack_diff_time_alert_second_attack(self) -> None :
        """ When we raise 1 alert on the second attack while we have 2 attacks at two times  
            but no blackout occur, we expect a reward of -1
            at step 3 (here with a window size of 2) and 1 step 4
            otherwise 0 at other time steps
            until the end of the episode where we have a bonus (here artificially 42)
        """
        kwargs_opponent = dict(lines_attacked=[ATTACKED_LINE]+['48_53_141'], 
                                   duration=[1,1], 
                                   steps_attack=[2, 3])
        with make(self.env_nm,
                  test=True,
                  difficulty="1", 
                  opponent_attack_cooldown=0, 
                  opponent_attack_duration=99999, 
                  opponent_budget_per_ts=1000, 
                  opponent_init_budget=10000., 
                  opponent_action_class=PlayableAction, 
                  opponent_class=TestOpponentMultiLines, 
                  kwargs_opponent=kwargs_opponent,
                  reward_class=AlertReward(reward_end_episode_bonus=42),
                  _add_to_name="_tarvnb2dtasa"
            ) as env : 
            env.seed(0)
            env.reset()
            step = 0
            for i in range(env.max_episode_duration()):
                act = env.action_space()
                if i == 2 : 
                    act = env.action_space({"raise_alert": [1]})
                obs, reward, done, info = env.step(act)
                step += 1
                if step in _get_steps_attack(kwargs_opponent, multi=True): 
                    assert info["opponent_attack_line"] is not None, f"no attack is detected at step {step}"
                else:
                    assert info["opponent_attack_line"]  is None, f"an attack is detected at step {step}"
                    
                if step == 4 : 
                    assert reward == 1, f"error for step {step}: {reward} vs 1"
                elif step == 5 : 
                    assert reward == -1, f"error for step {step}: {reward} vs -1"
                elif step == env.max_episode_duration(): 
                        assert reward == 42, f"error for step {step}: {reward} vs 42"
                else : 
                    assert reward == 0, f"error for step {step}: {reward} vs 0"


    def test_raise_illicit_alert(self) -> None:
        with make(
            self.env_nm,
            test=True,
            difficulty="1",
            reward_class=AlertReward(reward_end_episode_bonus=42)
        ) as env:
            env.seed(0)
            env.reset()
            assert type(env).dim_alerts == 10, f"dim_alerts: {type(env).dim_alerts} instead of 10"
            attackable_line_id = 10
            try : 
                act = env.action_space({"raise_alert": [attackable_line_id]})
            except Grid2OpException as exc_ : 
                assert exc_.args[0] == ('Impossible to modify the alert with your input. Please consult the '
                                        'documentation. The error was:\n"Grid2OpException IllegalAction '
                                        '"Impossible to change a raise alert id 10 because there are only '
                                        '10 on the grid (and in python id starts at 0)""')


class TestAlertBlackout(unittest.TestCase):
    """test the basic bahavior of the assistant alert feature when a blackout occur"""

    def setUp(self) -> None:
        self.env_nm = os.path.join(
            PATH_DATA_TEST, "l2rpn_idf_2023_with_alert"
        )
    
    def get_dn(self, env):
        return env.action_space({})

    def get_blackout(self, env):
        blackout_action = env.action_space({})
        blackout_action.gen_set_bus = [(0, -1)]
        return blackout_action

# Cas avec blackout 1 ligne attaquée
# return -10
    def test_assistant_reward_value_blackout_attack_no_alert(self) -> None :
        """
        When 1 line is attacked at step 2 and we don't raise any alert
        we expect a reward of -10 at step 3 
        """
        kwargs_opponent = dict(lines_attacked=[ATTACKED_LINE], 
                                   duration=3, 
                                   steps_attack=[2])
        with make(self.env_nm,
                  test=True,
                  difficulty="1", 
                  opponent_attack_cooldown=0, 
                  opponent_attack_duration=99999, 
                  opponent_budget_per_ts=1000, 
                  opponent_init_budget=10000., 
                  opponent_action_class=PlayableAction, 
                  opponent_class=TestOpponent, 
                  kwargs_opponent=kwargs_opponent, 
                  reward_class=AlertReward(reward_end_episode_bonus=42),
                  _add_to_name="_tarvbana"
            ) as env : 
            new_param = Parameters()
            new_param.MAX_LINE_STATUS_CHANGED = 10

            env.change_parameters(new_param)
            env.seed(0)
            env.reset()
            step = 0
            for i in range(env.max_episode_duration()):
                attackable_line_id = 0
                act = self.get_dn(env)
                if i == 3 :
                    act = self.get_blackout(env)
                obs, reward, done, info = env.step(act)
                step += 1
                
                if step in _get_steps_attack(kwargs_opponent): 
                    assert info["opponent_attack_line"] is not None, f"no attack is detected at step {step}"
                else:
                    assert info["opponent_attack_line"]  is None, f"an attack is detected at step {step}"
                
                if step == 4: 
                    # When the blackout occurs, reward is -10 because we didn't raise an attack
                    assert reward == -10, f"error for step {step}: {reward} vs -10"
                else : 
                    assert reward == 0, f"error for step {step}: {reward} vs 0"

                if done: 
                    break


# return 2
    def test_assistant_reward_value_blackout_attack_raise_good_alert(self) -> None :
        kwargs_opponent = dict(lines_attacked=[ATTACKED_LINE], 
                                   duration=3, 
                                   steps_attack=[2])
        with make(self.env_nm,
                  test=True,
                  difficulty="1", 
                  opponent_attack_cooldown=0, 
                  opponent_attack_duration=99999, 
                  opponent_budget_per_ts=1000, 
                  opponent_init_budget=10000., 
                  opponent_action_class=PlayableAction, 
                  opponent_class=TestOpponent, 
                  kwargs_opponent=kwargs_opponent,
                  _add_to_name="_tarvbarga"
            ) as env : 
            new_param = Parameters()
            new_param.MAX_LINE_STATUS_CHANGED = 10

            env.change_parameters(new_param)
            env.seed(0)
            env.reset()
            step = 0
            for i in range(env.max_episode_duration()):
                attackable_line_id = 0
                act = self.get_dn(env)
                if i == 3 : 
                    act = self.get_blackout(env)
                elif i == 1:
                    act = env.action_space({"raise_alert": [attackable_line_id]})
                obs, reward, done, info = env.step(act)
                step += 1
                if step in _get_steps_attack(kwargs_opponent): 
                    assert info["opponent_attack_line"] is not None, f"no attack is detected at step {step}"
                else:
                    assert info["opponent_attack_line"]  is None, f"an attack is detected at step {step}"
                
                if step == 4: 
                    assert reward == 2, f"error for step {step}: {reward} vs 2"
                else : 
                    assert reward == 0, f"error for step {step}: {reward} vs 0"

                if done: 
                    break


# return -10
    def test_assistant_reward_value_blackout_attack_raise_alert_just_before_blackout(self) -> None :
        """
        When 1 line is attacked at step 2 and we raise 1 alert  too late
        we expect a reward of -10 at step 3 
        """
        kwargs_opponent = dict(lines_attacked=[ATTACKED_LINE], 
                               duration=3, 
                               steps_attack=[2])
        with make(self.env_nm,
                  test=True,
                  difficulty="1", 
                  opponent_attack_cooldown=0, 
                  opponent_attack_duration=99999, 
                  opponent_budget_per_ts=1000, 
                  opponent_init_budget=10000., 
                  opponent_action_class=PlayableAction, 
                  opponent_class=TestOpponent, 
                  kwargs_opponent=kwargs_opponent,
                  _add_to_name="_tarvbarajbb"
            ) as env : 
            new_param = Parameters()
            new_param.MAX_LINE_STATUS_CHANGED = 10

            env.change_parameters(new_param)
            env.seed(0)
            env.reset()
            step = 0
            for i in range(env.max_episode_duration()):
                attackable_line_id = 0
                act = self.get_dn(env)
                if i == 3 : 
                    act = self.get_blackout(env)
                elif i == 2:
                    act = env.action_space({"raise_alert": [attackable_line_id]})
                obs, reward, done, info = env.step(act)
                step += 1
                if step in _get_steps_attack(kwargs_opponent): 
                    assert info["opponent_attack_line"] is not None, f"no attack is detected at step {step}"
                else:
                    assert info["opponent_attack_line"]  is None, f"an attack is detected at step {step}"
                
                if step == 4: 
                    assert reward == -10, f"error for step {step}: {reward} vs -10"
                else : 
                    assert reward == 0, f"error for step {step}: {reward} vs 0"

                if done: 
                    break
                
    def test_assistant_reward_value_blackout_attack_raise_alert_too_early(self) -> None :
        """
        When 1 line is attacked at step 2 and we raise 1 alert  too early
        we expect a reward of -10 at step 3 
        """
        # return -10
        kwargs_opponent = dict(lines_attacked=[ATTACKED_LINE], 
                                   duration=3, 
                                   steps_attack=[2])
        with make(self.env_nm,
                  test=True,
                  difficulty="1", 
                  opponent_attack_cooldown=0, 
                  opponent_attack_duration=99999, 
                  opponent_budget_per_ts=1000, 
                  opponent_init_budget=10000., 
                  opponent_action_class=PlayableAction, 
                  opponent_class=TestOpponent, 
                  kwargs_opponent=kwargs_opponent,
                  _add_to_name="_tarvbarate"
            ) as env : 
            new_param = Parameters()
            new_param.MAX_LINE_STATUS_CHANGED = 10

            env.change_parameters(new_param)
            env.seed(0)
            env.reset()
            step = 0
            for i in range(env.max_episode_duration()):
                attackable_line_id = 0
                act = self.get_dn(env)
                if i == 3 : 
                    act = self.get_blackout(env)
                elif i == 0:
                    act = env.action_space({"raise_alert": [attackable_line_id]})
                obs, reward, done, info = env.step(act)
                step += 1
                if step in _get_steps_attack(kwargs_opponent): 
                    assert info["opponent_attack_line"] is not None, f"no attack is detected at step {step}"
                else:
                    assert info["opponent_attack_line"]  is None, f"an attack is detected at step {step}"
                
                if step == 4: 
                    assert reward == -10, f"error for step {step}: {reward} vs -10"
                else : 
                    assert reward == 0, f"error for step {step}: {reward} vs 0"

                if done: 
                    break


# return 2
    def  test_assistant_reward_value_blackout_2_lines_same_step_in_window_good_alerts(self) -> None :
        """
        When 2 lines are attacked simustaneously at step 2 and we raise 2 alert 
        we expect a reward of 2 at step 3 
        """
        kwargs_opponent = dict(lines_attacked=[ATTACKED_LINE]+['48_53_141'], 
                               duration=3, 
                               steps_attack=[2,2])
        with make(self.env_nm,
                  test=True,
                  difficulty="1", 
                  opponent_attack_cooldown=0, 
                  opponent_attack_duration=99999, 
                  opponent_budget_per_ts=1000, 
                  opponent_init_budget=10000., 
                  opponent_action_class=PlayableAction, 
                  opponent_class=TestOpponent, 
                  kwargs_opponent=kwargs_opponent,
                  _add_to_name="_tarvb2lssiwga"
            ) as env : 
            new_param = Parameters()
            new_param.MAX_LINE_STATUS_CHANGED = 10

            env.change_parameters(new_param)
            env.seed(0)
            env.reset()
            step = 0
            for i in range(env.max_episode_duration()):
                attackable_line_id = 0
                act = self.get_dn(env)
                if i == 3 : 
                    act = self.get_blackout(env)
                elif i == 1:
                    act = env.action_space({"raise_alert": [0,1]})
                obs, reward, done, info = env.step(act)
                step += 1
                
                if step in _get_steps_attack(kwargs_opponent): 
                    assert info["opponent_attack_line"] is not None, f"no attack is detected at step {step}"
                else:
                    assert info["opponent_attack_line"]  is None, f"an attack is detected at step {step}"
                
                if step == 4: 
                    assert reward == 2, f"error for step {step}: {reward} vs 2"
                else : 
                    assert reward == 0, f"error for step {step}: {reward} vs 0"

                if done: 
                    break


# return -4
    def test_assistant_reward_value_blackout_2_lines_attacked_simulaneous_only_1_alert(self) -> None:
        """
        When 2 lines are attacked simustaneously at step 2 and we raise only 1 alert 
        we expect a reward of -4 at step 3 
        """
        kwargs_opponent = dict(lines_attacked=[ATTACKED_LINE]+['48_53_141'], 
                                   duration=3, 
                                   steps_attack=[2,2])
        with make(self.env_nm,
                  test=True,
                  difficulty="1", 
                  opponent_attack_cooldown=0, 
                  opponent_attack_duration=99999, 
                  opponent_budget_per_ts=1000, 
                  opponent_init_budget=10000., 
                  opponent_action_class=PlayableAction, 
                  opponent_class=TestOpponent, 
                  kwargs_opponent=kwargs_opponent,
                  _add_to_name="_tarvb2laso1a"
            ) as env : 
            new_param = Parameters()
            new_param.MAX_LINE_STATUS_CHANGED = 10

            env.change_parameters(new_param)
            env.seed(0)
            env.reset()
            step = 0
            for i in range(env.max_episode_duration()):
                attackable_line_id = 0
                act = self.get_dn(env)
                if i == 3 : 
                    act = self.get_blackout(env)
                elif i == 1:
                    act = env.action_space({"raise_alert": [0]})
                obs, reward, done, info = env.step(act)
                step += 1
                
                if step in _get_steps_attack(kwargs_opponent): 
                    assert info["opponent_attack_line"] is not None, f"no attack is detected at step {step}"
                else:
                    assert info["opponent_attack_line"]  is None, f"an attack is detected at step {step}"
                
                if step == 4: 
                    assert reward == -4, f"error for step {step}: {reward} vs -4"
                else : 
                    assert reward == 0, f"error for step {step}: {reward} vs 0"

                if done: 
                    break

# return 2
    def  test_assistant_reward_value_blackout_2_lines_different_step_in_window_good_alerts(self) -> None : 
        """
        When 2 lines are attacked at different steps 2 and 3 and we raise 2  alert 
        we expect a reward of 2 at step 3 
        """
        kwargs_opponent = dict(lines_attacked=[ATTACKED_LINE]+['48_53_141'], 
                               duration=[1,1], 
                               steps_attack=[2, 3])
        with make(self.env_nm,
                  test=True,
                  difficulty="1", 
                  opponent_attack_cooldown=0, 
                  opponent_attack_duration=99999, 
                  opponent_budget_per_ts=1000, 
                  opponent_init_budget=10000., 
                  opponent_action_class=PlayableAction, 
                  opponent_class=TestOpponentMultiLines, 
                  kwargs_opponent=kwargs_opponent,
                  reward_class=AlertReward(reward_end_episode_bonus=42),
                  _add_to_name="_tarvb2ldsiwga"
            ) as env : 
            env.seed(0)
            env.reset()
            step = 0
            for i in range(env.max_episode_duration()):
                act = self.get_dn(env)
                if i == 1 :
                    act = env.action_space({"raise_alert": [0]})
                elif i == 2 : 
                    act = env.action_space({"raise_alert": [1]})
                elif i == 3 : 
                    act = self.get_blackout(env)
                obs, reward, done, info = env.step(act)
                step += 1
                
                if step in _get_steps_attack(kwargs_opponent, multi=True): 
                    assert info["opponent_attack_line"] is not None, f"no attack is detected at step {step}"
                else:
                    assert info["opponent_attack_line"]  is None, f"an attack is detected at step {step}"
                
                if step == 4 : 
                    assert reward == 2, f"error for step {step}: {reward} vs 2"
                elif step == env.max_episode_duration(): 
                        assert reward == 42, f"error for step {step}: {reward} vs 42"
                else : 
                    assert reward == 0, f"error for step {step}: {reward} vs 0"

                if done : 
                    break

    def test_assistant_reward_value_blackout_2_lines_attacked_different_step_in_window_only_1_alert_on_first_attacked_line(self) -> None:
        """
        When 2 lines are attacked at different steps 2 and 3 and we raise 1 alert on the first attack
        we expect a reward of -4 at step 3 
        """
        kwargs_opponent = dict(lines_attacked=[ATTACKED_LINE]+['48_53_141'], 
                               duration=[1,1], 
                               steps_attack=[2, 3])
        with make(self.env_nm,
                  test=True,
                  difficulty="1", 
                  opponent_attack_cooldown=0, 
                  opponent_attack_duration=99999, 
                  opponent_budget_per_ts=1000, 
                  opponent_init_budget=10000., 
                  opponent_action_class=PlayableAction, 
                  opponent_class=TestOpponentMultiLines, 
                  kwargs_opponent=kwargs_opponent,
                  reward_class=AlertReward(reward_end_episode_bonus=42),
                  _add_to_name="_tarvb2ladsiwo1aofal"
            ) as env : 
            env.seed(0)
            env.reset()
            step = 0
            for i in range(env.max_episode_duration()):
                act = self.get_dn(env)
                if i == 1 :
                    act = env.action_space({"raise_alert": [0]})
                elif i == 3 : 
                    act = self.get_blackout(env)
                obs, reward, done, info = env.step(act)
                step += 1
                if step in _get_steps_attack(kwargs_opponent, multi=True): 
                    assert info["opponent_attack_line"] is not None, f"no attack is detected at step {step}"
                else:
                    assert info["opponent_attack_line"]  is None, f"an attack is detected at step {step}"
                    
                if step == 4 : 
                    assert reward == -4, f"error for step {step}: {reward} vs -4"
                elif step == env.max_episode_duration(): 
                        assert reward == 42, f"error for step {step}: {reward} vs 42"
                else : 
                    assert reward == 0, f"error for step {step}: {reward} vs 0"
            
                if done : 
                    break

# return -4
    def test_assistant_reward_value_blackout_2_lines_attacked_different_step_in_window_only_1_alert_on_second_attacked_line(self) -> None:
        """
        When 2 lines are attacked at different steps 2 and 3 and we raise 1 alert on the second attack
        we expect a reward of -4 at step 3 
        """
        kwargs_opponent = dict(lines_attacked=[ATTACKED_LINE]+['48_53_141'], 
                               duration=[1,1], 
                               steps_attack=[2, 3])
        with make(self.env_nm,
                  test=True,
                  difficulty="1", 
                  opponent_attack_cooldown=0, 
                  opponent_attack_duration=99999, 
                  opponent_budget_per_ts=1000, 
                  opponent_init_budget=10000., 
                  opponent_action_class=PlayableAction, 
                  opponent_class=TestOpponentMultiLines, 
                  kwargs_opponent=kwargs_opponent,
                  reward_class=AlertReward(reward_end_episode_bonus=42),
                  _add_to_name="_tarvb2ladsiwo1aosal"
            ) as env : 
            env.seed(0)
            env.reset()
            step = 0
            for i in range(env.max_episode_duration()):
                act = self.get_dn(env)
                if i == 2 :
                    act = env.action_space({"raise_alert": [1]})
                elif i == 3 : 
                    act = self.get_blackout(env)
                obs, reward, done, info = env.step(act)
                step += 1
                if step in _get_steps_attack(kwargs_opponent, multi=True): 
                    assert info["opponent_attack_line"] is not None, f"no attack is detected at step {step}"
                else:
                    assert info["opponent_attack_line"]  is None, f"an attack is detected at step {step}"
                    
                if step == 4 : 
                    assert reward == -4., f"error for step {step}: {reward} vs -4"
                else : 
                    assert reward == 0, f"error for step {step}: {reward} vs 0"
            
                if done : 
                    break

# return 2 
    def test_assistant_reward_value_blackout_2_lines_attacked_different_1_in_window_1_good_alert(self) -> None:
        """
        When 2 lines are attacked at different steps 2 and 4 and we raise 1 alert on the second attack
        we expect a reward of 2 at step 5 
        """
        kwargs_opponent = dict(lines_attacked=[ATTACKED_LINE]+['48_53_141'], 
                               duration=[1, 1], 
                               steps_attack=[2, 5])
        with make(self.env_nm,
                  test=True,
                  difficulty="1", 
                  opponent_attack_cooldown=0, 
                  opponent_attack_duration=99999, 
                  opponent_budget_per_ts=1000, 
                  opponent_init_budget=10000., 
                  opponent_action_class=PlayableAction, 
                  opponent_class=TestOpponentMultiLines, 
                  kwargs_opponent=kwargs_opponent,
                  reward_class=AlertReward(reward_end_episode_bonus=42),
                  _add_to_name="_tarvb2lad1iw1ga"
            ) as env : 
            env.seed(0)
            env.reset()
            step = 0
            for i in range(env.max_episode_duration()):
                act = self.get_dn(env)
                if i == 4 :
                    act = env.action_space({"raise_alert": [1]})
                elif i == 6 : 
                    act = self.get_blackout(env)
                obs, reward, done, info = env.step(act)
                step += 1

                if step in _get_steps_attack(kwargs_opponent, multi=True): 
                    assert info["opponent_attack_line"] is not None, f"no attack is detected at step {step}"
                else:
                    assert info["opponent_attack_line"]  is None, f"an attack is detected at step {step}"
                
                if step == 4 : 
                    assert reward == 1, f"error for step {step}: {reward} vs 1"
                elif step == 7 : 
                    assert reward == 2, f"error for step {step}: {reward} vs 2"
                else : 
                    assert reward == 0, f"error for step {step}: {reward} vs 0"
            
                if done : 
                    break

# return 0 
    def test_assistant_reward_value_blackout_no_attack_alert(self) -> None :

        """Even if there is a blackout, an we raise an alert
           we expect a reward of 0 because there is no attack"""
        with make(
            self.env_nm,
            test=True,
            difficulty="1",
            reward_class=AlertReward(reward_end_episode_bonus=42)
        ) as env:
            env.seed(0)
            env.reset()

            done = False
            for i in range(env.max_episode_duration()):
                act = self.get_dn(env)
                if i == 3 : 
                    act = self.get_blackout(env)
                elif i == 1:
                    act = env.action_space({"raise_alert": [0]})
                obs, reward, done, info = env.step(act)
                if info["opponent_attack_line"] is None : 
                    assert reward == 0.
                else : 
                    raise Grid2OpException('No attack expected')

                if done : 
                    break
            
            assert done

# return 0 
    def test_assistant_reward_value_blackout_no_attack_no_alert(self) -> None :
        """Even if there is a blackout, an we don't raise an alert
           we expect a reward of 0 because there is no attack"""
        with make(
            self.env_nm,
            test=True,
            difficulty="1",
            reward_class=AlertReward(reward_end_episode_bonus=42)
        ) as env:
            env.seed(0)
            env.reset()

            done = False
            for i in range(env.max_episode_duration()):
                act = self.get_dn(env)
                if i == 3 : 
                    act = self.get_blackout(env)
                obs, reward, done, info = env.step(act)
                if info["opponent_attack_line"] is None : 
                    assert reward == 0.
                else : 
                    raise Grid2OpException('No attack expected')

                if done : 
                    break
            
            assert done

# return 0 
    def test_assistant_reward_value_blackout_attack_before_window_alert(self) -> None :
        """Even if there is a blackout, an we raise an alert too early
           we expect a reward of 0 because there is no attack"""
        with make(
            self.env_nm,
            test=True,
            difficulty="1",
            reward_class=AlertReward(reward_end_episode_bonus=42)
        ) as env:
            env.seed(0)
            env.reset()

            done = False
            for i in range(env.max_episode_duration()):
                act = self.get_dn(env)
                if i == 3 : 
                    act = self.get_blackout(env)
                elif i == 0:
                    act = env.action_space({"raise_alert": [0]})
                obs, reward, done, info = env.step(act)
                if info["opponent_attack_line"] is None : 
                    assert reward == 0.
                else : 
                    raise Grid2OpException('No attack expected')

                if done : 
                    break
            
            assert done

# return 0 
    def test_assistant_reward_value_blackout_attack_before_window_no_alert(self) -> None :
        """Even if there is a blackout, an we raise an alert too late
           we expect a reward of 0 because there is no attack"""
        with make(
            self.env_nm,
            test=True,
            difficulty="1",
            reward_class=AlertReward(reward_end_episode_bonus=42)
        ) as env:
            env.seed(0)
            env.reset()

            done = False
            for i in range(env.max_episode_duration()):
                act = self.get_dn(env)
                if i == 3 : 
                    act = self.get_blackout(env)
                elif i == 4:
                    act = env.action_space({"raise_alert": [0]})
                obs, reward, done, info = env.step(act)
                
                if info["opponent_attack_line"] is None : 
                    assert reward == 0.
                else : 
                    raise Grid2OpException('No attack expected')

                if done : 
                    break
            
            assert done

# TODO : test des actions ambigues  
# Action ambigue : par exemple alert sur la ligne (nb_lignes)+1 
# Aller voir la doc : file:///home/crochepierrelau/Documents/Git/Grid2Op/documentation/html/action.html#illegal-vs-ambiguous

# TODO : test runner

# TODO test simulate

# TODO test get_forecast_env

