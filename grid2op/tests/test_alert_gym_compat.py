# Copyright (c) 2019-2023, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import os
import unittest
import numpy as np


from grid2op.tests.helper_path_test import PATH_DATA_TEST

import grid2op
from grid2op.dtypes import dt_int
from grid2op.Action import PlayableAction

try:
    from grid2op.gym_compat import GymEnv, GYMNASIUM_AVAILABLE
    from grid2op.gym_compat import MultiToTupleConverter
    from grid2op.gym_compat import (
        BoxGymObsSpace,
        BoxGymActSpace,
        MultiDiscreteActSpace,
    )

    CAN_DO_TEST = True
except ImportError:
    CAN_DO_TEST = False

if CAN_DO_TEST:
    if GYMNASIUM_AVAILABLE:
        import gymnasium as gym_for_test_agc  # type: ignore
    else:
        import gym as gym_for_test_agc  # type: ignore

import pdb

import warnings

warnings.simplefilter("error")


class TestGymAlertCompat(unittest.TestCase):
    def _skip_if_no_gym(self):
        if not CAN_DO_TEST:
            self.skipTest("Gym is not available")

    def setUp(self) -> None:
        self._skip_if_no_gym()

        self.env_nm = os.path.join(PATH_DATA_TEST, "l2rpn_idf_2023_with_alert")

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make(
                self.env_nm, test=True, _add_to_name=type(self).__name__
            )
        self.env.seed(0)
        self.env.reset()  # seed part !

    def tearDown(self) -> None:
        self.env.close()

    def test_print_alert(self):
        env_gym = GymEnv(self.env)
        str_ = env_gym.action_space.__str__()

        assert str_ == (
            "Dict('change_bus': MultiBinary(177), 'change_line_status': MultiBinary(59), "
            "'curtail': Box([-1.  0. -1. -1. -1.  0.  0.  0.  0.  0. -1.  0.  0. -1.  0.  0. -1.  "
            "0.\n  0. -1. -1. -1.], [-1.  1. -1. -1. -1.  1.  1.  1.  1.  1. -1.  1.  1. -1.  1.  "
            "1. -1.  1.\n  1. -1. -1. -1.], (22,), float32), 'raise_alert': MultiBinary(10), "
            "'redispatch': Box([ -1.4   0.   -1.4 -10.4  -1.4   0.    0.    0.    0.    0.   -2.8   "
            "0.\n   0.   -2.8   0.    0.   -4.3   0.    0.   -2.8  -8.5  -9.9], [ 1.4  0.   1.4 10.4  "
            "1.4  0.   0.   0.   0.   0.   2.8  0.   0.   2.8\n  0.   0.   4.3  0.   0.   2.8  8.5  9.9], "
            "(22,), float32), 'set_bus': Box(-1, 2, (177,), int32), 'set_line_status': Box(-1, 1, (59,), int32))"
        )
        str_ = env_gym.observation_space.__str__()
        act = self.env.action_space()
        act.raise_alert = [2]
        act_gym = env_gym.action_space.to_gym(act)
        act_str = act_gym.__str__()
        assert act_str == (
            "OrderedDict({'change_bus': array([False, False, False, False, False, "
            "False, False, False, False,\n       False, False, False, False, False, "
            "False, False, False, False,\n       False, False, False, False, False, "
            "False, False, False, False,\n       False, False, False, False, False, "
            "False, False, False, False,\n       False, False, False, False, False, "
            "False, False, False, False,\n       False, False, False, False, False, "
            "False, False, False, False,\n       False, False, False, False, False, "
            "False, False, False, False,\n       False, False, False, False, False, "
            "False, False, False, False,\n       False, False, False, False, False, "
            "False, False, False, False,\n       False, False, False, False, False, "
            "False, False, False, False,\n       False, False, False, False, False, "
            "False, False, False, False,\n       False, False, False, False, False, "
            "False, False, False, False,\n       False, False, False, False, False, "
            "False, False, False, False,\n       False, False, False, False, False, "
            "False, False, False, False,\n       False, False, False, False, False, "
            "False, False, False, False,\n       False, False, False, False, False, "
            "False, False, False, False,\n       False, False, False, False, False, "
            "False, False, False, False,\n       False, False, False, False, False, "
            "False, False, False, False,\n       False, False, False, False, False, "
            "False, False, False, False,\n       False, False, False, False, False, "
            "False]), 'change_line_status': array([False, False, False, False, False"
            ", False, False, False, False,\n       False, False, False, False, False"
            ", False, False, False, False,\n       False, False, False, False, False"
            ", False, False, False, False,\n       False, False, False, False, False"
            ", False, False, False, False,\n       False, False, False, False, False"
            ", False, False, False, False,\n       False, False, False, False, False"
            ", False, False, False, False,\n       False, False, False, False, False"
            "]), 'curtail': array([-1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,"
            " -1., -1., -1.,\n       -1., -1., -1., -1., -1., -1., -1., -1., -1.], "
            "dtype=float32), 'raise_alert': array([False, False,  True, False, "
            "False, False, False, False, False,\n       False]), 'redispatch': "
            "array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., "
            "0.,\n       0., 0., 0., 0., 0.], dtype=float32), 'set_bus': array([0, "
            "0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,\n       "
            "0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,\n   "
            "    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,"
            "\n       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, "
            "0,\n       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, "
            "0, 0,\n       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, "
            "0, 0, 0,\n       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, "
            "0, 0, 0, 0,\n       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, "
            "0, 0, 0, 0, 0,\n       0], dtype=int32), 'set_line_status': array([0, "
            "0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,\n    "
            "   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, "
            "0,\n       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=int32)})"
        )

    def test_convert_alert_to_gym(self):
        """test i can create the env"""
        env_gym = GymEnv(self.env)
        dim_act_space = np.sum(
            [
                np.sum(env_gym.action_space[el].shape)
                for el in env_gym.action_space.spaces
            ]
        )
        assert dim_act_space == 526, f"{dim_act_space} != 526"
        dim_obs_space = np.sum(
            [
                np.sum(env_gym.observation_space[el].shape).astype(int)
                for el in env_gym.observation_space.spaces
            ]
        )
        size_th = 1718  # as of grid2Op 1.9.1 (where alerts are added)
        size_th = 1740  # as of grid2Op 1.11.0 (where gen_p_delta)
        size_th = 1799  # as of grid2Op 1.11.0 (where timestep_protection_engaged)
        assert dim_obs_space == size_th, (
            f"Size should be {size_th} but is {dim_obs_space}"
        )

        # test that i can do basic stuff there
        obs, info = env_gym.reset()
        for k in env_gym.observation_space.spaces.keys():
            assert obs[k] in env_gym.observation_space[k], f"error for key: {k}"
        act = env_gym.action_space.sample()
        obs2, reward2, done2, truncated, info2 = env_gym.step(act)
        assert obs2 in env_gym.observation_space

        # test for the __str__ method (it could crash)
        _ = self.env.action_space.__str__()
        _ = self.env.observation_space.__str__()

    def test_ignore_some_alert_attributes(self):
        """test the ignore_attr method"""
        env_gym = GymEnv(self.env)
        env_gym.action_space = env_gym.action_space.ignore_attr(
            "last_alert"
        ).ignore_attr("was_alert_used_after_attack")
        dim_act_space = np.sum(
            [
                np.sum(env_gym.action_space[el].shape)
                for el in env_gym.action_space.spaces
            ]
        )
        assert dim_act_space == 526, f"{dim_act_space=} != 526"

    def test_keep_only_2_alert_attr(self):
        """test the keep_only_attr method"""
        env_gym = GymEnv(self.env)
        env_gym.observation_space = env_gym.observation_space.keep_only_attr(
            ["last_alert", "was_alert_used_after_attack"]
        )
        new_dim_obs_space = np.sum(
            [
                np.sum(env_gym.observation_space[el].shape).astype(int)
                for el in env_gym.observation_space.spaces
            ]
        )
        assert new_dim_obs_space == 10, f"{new_dim_obs_space=} != 10"

    def test_all_together_in_alert(self):
        """combine all test above (for the action space)"""
        env_gym = GymEnv(self.env)
        env_gym.action_space = env_gym.action_space.reencode_space(
            "raise_alert", MultiToTupleConverter()
        )

        assert isinstance(
            env_gym.action_space["raise_alert"], gym_for_test_agc.spaces.Tuple
        )

        act_gym = env_gym.action_space.sample()
        act_glop = env_gym.action_space.from_gym(act_gym)
        act_gym2 = env_gym.action_space.to_gym(act_glop)
        act_glop2 = env_gym.action_space.from_gym(act_gym2)

        assert act_gym in env_gym.action_space
        assert act_gym2 in env_gym.action_space

        assert isinstance(act_gym["raise_alert"], tuple)

        # check the gym actions are the same
        for k in act_gym.keys():
            assert np.array_equal(act_gym[k], act_gym2[k]), f"error for {k}"
        for k in act_gym2.keys():
            assert np.array_equal(act_gym[k], act_gym2[k]), f"error for {k}"
        # check grid2op action are the same
        assert act_glop == act_glop2

    def test_low_high_alert_obs_space(self):
        """test the observation space, by default, is properly converted"""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = grid2op.make(
                "l2rpn_idf_2023", test=True, _add_to_name=type(self).__name__
            )
        env.seed(0)
        env.reset()  # seed part !
        env_gym = GymEnv(env)

        for key, low, high in zip(
            [
                "attack_under_alert",
                "time_since_last_alert",
                "alert_duration",
                "time_since_last_attack",
                "was_alert_used_after_attack",
            ],
            [
                np.zeros(env.dim_alerts, dtype=dt_int) - 1,
                np.zeros(env.dim_alerts, dtype=dt_int) - 1,
                np.zeros(env.dim_alerts, dtype=dt_int),
                np.zeros(env.dim_alerts, dtype=dt_int) - 1,
                np.zeros(env.dim_alerts, dtype=dt_int) - 1,
            ],
            [
                np.zeros(env.dim_alerts, dtype=dt_int) + 1,
                np.zeros(env.dim_alerts, dtype=dt_int) + 2147483647,
                np.zeros(env.dim_alerts, dtype=dt_int) + 2147483647,
                np.zeros(env.dim_alerts, dtype=dt_int) + 2147483647,
                np.zeros(env.dim_alerts, dtype=dt_int) + 1,
            ],
        ):
            assert key in env_gym.observation_space.spaces
            assert isinstance(
                env_gym.observation_space[key], gym_for_test_agc.spaces.Box
            )
            assert env_gym.observation_space[key].shape == (env.dim_alerts,)
            assert env_gym.observation_space[key].dtype == dt_int

            assert np.array_equal(env_gym.observation_space[key].low, low), (
                f"issue for {key}"
            )
            assert np.array_equal(env_gym.observation_space[key].high, high), (
                f"issue for {key}"
            )

        key = "total_number_of_alert"
        assert key in env_gym.observation_space.spaces
        assert isinstance(env_gym.observation_space[key], gym_for_test_agc.spaces.Box)
        assert env_gym.observation_space[key].dtype == dt_int
        assert env_gym.observation_space[key].shape == (1,)
        assert env_gym.observation_space.spaces[key].low == [0]
        assert env_gym.observation_space.spaces[key].high == [2147483647]

        key = "active_alert"
        assert key in env_gym.observation_space.spaces
        assert isinstance(
            env_gym.observation_space[key], gym_for_test_agc.spaces.MultiBinary
        )
        assert env_gym.observation_space[key].shape == (22,)


class TestBoxGymObsSpaceWithAlert(unittest.TestCase):
    def _skip_if_no_gym(self):
        if not CAN_DO_TEST:
            self.skipTest("Gym is not available")

    def setUp(self) -> None:
        self._skip_if_no_gym()

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make(
                "l2rpn_idf_2023", test=True, _add_to_name=type(self).__name__
            )
        self.env.seed(0)
        self.env.reset()  # seed part !
        self.obs_env = self.env.reset()
        self.env_gym = GymEnv(self.env)

    def test_can_create(self):
        kept_attr = [
            "active_alert",
            "alert_duration",
            "attack_under_alert",
            "time_since_last_alert",
            "time_since_last_attack",
            "total_number_of_alert",
            "was_alert_used_after_attack",
        ]

        self.env_gym.observation_space = BoxGymObsSpace(
            self.env.observation_space,
            attr_to_keep=kept_attr,
        )
        obs_gym, info = self.env_gym.reset()
        assert self.env_gym.observation_space._attr_to_keep == sorted(kept_attr)
        assert len(obs_gym) == 133, f"{len(obs_gym)} vs 133"
        assert self.env_gym.observation_space.dtype == dt_int
        assert obs_gym in self.env_gym.observation_space


class TestAllGymActSpaceWithAlert(unittest.TestCase):
    def _skip_if_no_gym(self):
        if not CAN_DO_TEST:
            self.skipTest("Gym is not available")

    def setUp(self) -> None:
        self._skip_if_no_gym()

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make(
                os.path.join(PATH_DATA_TEST, "l2rpn_idf_2023_with_alert"),
                test=True,
                action_class=PlayableAction,
                _add_to_name=type(self).__name__,
            )
        self.env.seed(0)
        self.env.reset()  # seed part !
        self.obs_env = self.env.reset()
        self.env_gym = GymEnv(self.env)

    def test_supported_keys_box(self):
        """test all the attribute of the action can be modified when the action is converted to a float"""
        all_attr = {
            "raise_alert": len(self.env.alertable_line_ids),
        }
        func_check = {
            "raise_alert": lambda act: np.any(act.raise_alert)
            and ~np.all(act.raise_alert),
        }

        for attr_nm in sorted(all_attr.keys()):
            kept_attr = [attr_nm]
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                self.env_gym.action_space = BoxGymActSpace(
                    self.env.action_space, attr_to_keep=kept_attr
                )
            self.env_gym.action_space.seed(0)
            gym_act = self.env_gym.action_space.sample()
            grid2op_act = self.env_gym.action_space.from_gym(gym_act)
            assert isinstance(grid2op_act, PlayableAction)
            assert self.env_gym.action_space._attr_to_keep == kept_attr
            assert len(self.env_gym.action_space.sample()) == all_attr[attr_nm], (
                f"wrong size for {attr_nm}"
            )
            # check that all types
            ok_ = func_check[attr_nm](grid2op_act)
            if not ok_ and attr_nm != "set_storage":
                # NB for "set_storage" as there are no storage unit on this grid, then this test is doomed to fail
                # this is why i don't perform it in this case
                raise RuntimeError(
                    f"Some property of the actions are not modified for attr {attr_nm}"
                )

    def test_supported_keys_multidiscrete(self):
        """test that i can modify every action with the keys"""
        dims = {
            "raise_alert": len(self.env.alertable_line_ids),
        }
        func_check = {
            "raise_alert": lambda act: np.any(act.raise_alert)
            and ~np.all(act.raise_alert),
        }

        for attr_nm in sorted(dims.keys()):
            kept_attr = [attr_nm]
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                self.env_gym.action_space = MultiDiscreteActSpace(
                    self.env.action_space, attr_to_keep=kept_attr
                )
            assert self.env_gym.action_space._attr_to_keep == kept_attr
            self.env_gym.action_space.seed(0)
            assert len(self.env_gym.action_space.sample()) == dims[attr_nm], (
                f"wrong size for {attr_nm}"
            )
            grid2op_act = self.env_gym.action_space.from_gym(
                self.env_gym.action_space.sample()
            )
            assert isinstance(grid2op_act, PlayableAction)
            # check that all types
            ok_ = func_check[attr_nm](grid2op_act)
            if not ok_ and attr_nm != "set_storage":
                # NB for "set_storage" as there are no storage unit on this grid, then this test is doomed to fail
                # this is why i don't perform it in this case
                raise RuntimeError(
                    f"Some property of the actions are not modified for attr {attr_nm}"
                )


class ObsAlertAttr(unittest.TestCase):
    def test_alert_attr_in_obs(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = grid2op.make(
                "l2rpn_idf_2023",
                test=True,
                action_class=PlayableAction,
                _add_to_name=type(self).__name__,
            )
        gym_env = GymEnv(env)
        obs, info = gym_env.reset()
        alert_attrs = [
            "active_alert",
            "alert_duration",
            "attack_under_alert",
            "time_since_last_alert",
            "time_since_last_attack",
            "total_number_of_alert",
            "was_alert_used_after_attack",
        ]
        for el in alert_attrs:
            assert el in obs.keys(), f'"{el}" not in obs.keys()'


if __name__ == "__main__":
    unittest.main()
