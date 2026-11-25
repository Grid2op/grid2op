# Copyright (c) 2026, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import unittest
import warnings
import numpy as np

import grid2op
from grid2op.Action import CompleteAction, ShuntAction
from grid2op.Backend.educPandaPowerBackend import EducPandaPowerBackend
from grid2op.dtypes import dt_float, dt_int
from grid2op.Exceptions import AmbiguousAction, IllegalAction
from grid2op.gym_compat import BoxGymActSpace, BoxGymObsSpace, GymEnv
from grid2op.Rules import AlwaysLegal


class TestShuntActProperties(unittest.TestCase):
    """test the `shunt_p`, `shunt_q` and `shunt_bus` properties of the actions"""

    def setUp(self) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make("l2rpn_case14_sandbox",
                                    test=True,
                                    action_class=CompleteAction,
                                    _add_to_name=type(self).__name__)

    def tearDown(self) -> None:
        self.env.close()

    def test_shunt_p(self):
        act = self.env.action_space()
        act.shunt_p = [(0, 1.0)]
        assert act._modif_shunt
        assert act.shunt_p[0] == 1.0
        assert act.as_dict()["shunt"]["shunt_p"][0] == 1.0

    def test_shunt_q(self):
        act = self.env.action_space()
        act.shunt_q = [(0, -25.0)]
        assert act._modif_shunt
        assert act.shunt_q[0] == -25.0
        assert act.as_dict()["shunt"]["shunt_q"][0] == -25.0

    def test_shunt_bus(self):
        act = self.env.action_space()
        act.shunt_bus = [(0, -1)]
        assert act._modif_shunt
        assert act.shunt_bus[0] == -1

    def test_same_as_dict(self):
        """the properties and the "shunt" key of the dictionary give the same action"""
        act_prop = self.env.action_space()
        act_prop.shunt_q = [(0, -25.0)]
        act_dict = self.env.action_space({"shunt": {"shunt_q": [(0, -25.0)]}})
        assert act_prop == act_dict

    def test_eq_shunt_bus(self):
        """two actions setting the shunts to different buses are not equal"""
        act_1 = self.env.action_space()
        act_1.shunt_bus = [(0, 1)]
        act_2 = self.env.action_space()
        act_2.shunt_bus = [(0, -1)]
        assert not (act_1 == act_2)
        assert not (act_2 == act_1)
        act_3 = self.env.action_space()
        act_3.shunt_bus = [(0, 1)]
        assert act_1 == act_3

    def test_to_vect_refreshed(self):
        """setting a shunt property after `to_vect` updates the vector"""
        for prop_nm, val in [("shunt_p", [(0, 1.0)]),
                             ("shunt_q", [(0, -25.0)]),
                             ("shunt_bus", [(0, -1)])]:
            act = self.env.action_space()
            act.to_vect()  # the vector is computed (and cached) before the modification
            setattr(act, prop_nm, val)
            ref = self.env.action_space()
            setattr(ref, prop_nm, val)
            assert np.array_equal(act.to_vect(), ref.to_vect(), equal_nan=True), f"error for {prop_nm}"

    def test_wrong_input(self):
        """a wrong input raises and does not modify the action"""
        act = self.env.action_space()
        act.shunt_q = [(0, -25.0)]
        n_shunt = type(self.env).n_shunt
        with self.assertRaises(AmbiguousAction):
            act.shunt_q = [(n_shunt, -10.0)]
        assert act.shunt_q[0] == -25.0
        with self.assertRaises(AmbiguousAction):
            act.shunt_bus = [(0, type(self.env).n_busbar_per_sub + 1)]
        assert act.shunt_bus[0] == 0

    def test_wrong_input_message(self):
        """the error message names the element (a shunt), not the property"""
        act = self.env.action_space()
        n_shunt = type(self.env).n_shunt
        for prop_nm, val in [("shunt_q", -10.0), ("shunt_bus", 1)]:
            with self.assertRaises(AmbiguousAction) as ctx:
                setattr(act, prop_nm, [(n_shunt, val)])
            msg = str(ctx.exception)
            assert f"Impossible to set a shunt id {n_shunt}" in msg, f"error for {prop_nm}: {msg}"

    def test_property_read_only(self):
        act = self.env.action_space()
        with self.assertRaises(ValueError):
            act.shunt_q[0] = -25.0

    def test_str(self):
        """`print(act)` shows the shunt modifications, and only when there are some"""
        name_sh = type(self.env).name_shunt[0]
        act = self.env.action_space()
        assert "shunt" not in str(act)
        act.shunt_q = [(0, -25.0)]
        str_act = str(act)
        assert "Modify the shunts in the following way:" in str_act
        assert f'For shunt "{name_sh}": set its reactive power to -25.00 MVAr' in str_act
        act = self.env.action_space()
        act.shunt_bus = [(0, -1)]
        assert f'For shunt "{name_sh}": disconnect it' in str(act)
        act = self.env.action_space()
        act.shunt_p = [(0, 1.0)]
        act.shunt_bus = [(0, 1)]
        assert f'For shunt "{name_sh}": connect it to busbar 1, set its active power to 1.00 MW' in str(act)

    def _aux_step_shunt_q(self, shunt_q):
        """always play the same scenario, so that the runs are comparable"""
        self.env.set_id(0)
        self.env.reset()
        act = self.env.action_space()
        if shunt_q is not None:
            act.shunt_q = [(0, shunt_q)]
        obs, *_ = self.env.step(act)
        return obs

    def test_shunt_q_affects_voltage(self):
        """a capacitive shunt should raise the voltages, an inductive one lower them"""
        v_ref = 1.0 * self._aux_step_shunt_q(None).load_v
        v_capa = self._aux_step_shunt_q(-30.0).load_v
        v_indu = self._aux_step_shunt_q(30.0).load_v

        assert (v_capa >= v_ref).all()
        assert (v_indu <= v_ref).all()
        assert (v_capa > v_ref).any()
        assert (v_indu < v_ref).any()

    def test_not_authorized(self):
        """the default (playable) action type does not allow to modify the shunts"""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = grid2op.make("l2rpn_case14_sandbox",
                               test=True,
                               _add_to_name=type(self).__name__)
        try:
            act = env.action_space()
            with self.assertRaises(IllegalAction):
                act.shunt_q = [(0, -25.0)]
        finally:
            env.close()


class TestShuntAction(unittest.TestCase):
    """test the :class:`ShuntAction` action type"""

    def setUp(self) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make("l2rpn_case14_sandbox",
                                    test=True,
                                    action_class=ShuntAction,
                                    _add_to_name=type(self).__name__)

    def tearDown(self) -> None:
        self.env.close()

    def test_only_shunts(self):
        act_cls = self.env.action_space.actionClass
        assert issubclass(act_cls, ShuntAction)
        assert act_cls.authorized_keys == {"shunt"}
        assert self.env.action_space.n == 3 * type(self.env).n_shunt

    def test_other_levers_refused(self):
        act = self.env.action_space()
        with self.assertRaises(IllegalAction):
            act.redispatch = [(0, 1.0)]
        with self.assertRaises(IllegalAction):
            act.line_set_status = [(0, -1)]
        with self.assertRaises(AmbiguousAction):
            act.load_set_bus = [(0, 2)]

    def test_same_as_dict(self):
        act_prop = self.env.action_space()
        act_prop.shunt_q = [(0, -25.0)]
        act_dict = self.env.action_space({"shunt": {"shunt_q": [(0, -25.0)]}})
        assert act_prop == act_dict

    def test_vect_roundtrip(self):
        act = self.env.action_space()
        act.shunt_q = [(0, -25.0)]
        act.shunt_bus = [(0, 1)]
        vect = act.to_vect()
        assert vect.shape == (self.env.action_space.n,)
        act2 = self.env.action_space.from_vect(vect)
        assert act == act2
        assert act2.shunt_q[0] == -25.0
        assert act2.shunt_bus[0] == 1

    def test_sample(self):
        self.env.action_space.seed(0)
        act = self.env.action_space.sample()
        assert isinstance(act, ShuntAction)
        # sampling shunt actions is not supported yet: the sampled action does nothing
        assert not act.can_affect_something()

    def test_step(self):
        self.env.set_id(0)
        self.env.reset()
        act = self.env.action_space()
        act.shunt_q = [(0, -25.0)]
        obs, reward, done, info = self.env.step(act)
        assert not done
        assert not info["is_illegal"]
        assert not info["is_ambiguous"]
        # a shunt is modeled with a fixed susceptance in pandapower: the reactive power
        # actually exchanged scales with the square of the voltage, hence the inequality
        assert obs._shunt_q[0] < -25.0


class TestShuntBusSeveralShunts(unittest.TestCase):
    """change the bus of some (more than one, but not all) shunts of a grid with several shunts"""

    def setUp(self) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make("l2rpn_idf_2023",
                                    test=True,
                                    action_class=ShuntAction,
                                    gamerules_class=AlwaysLegal,  # only the backend is tested here
                                    _add_to_name=type(self).__name__)
        self.env.set_id(0)
        self.env.reset(seed=0)

    def tearDown(self) -> None:
        self.env.close()

    def test_shunt_bus_several_shunts(self):
        # all the shunts of this grid start disconnected: connect 2 of them
        act = self.env.action_space()
        act.shunt_bus = [(0, 1), (1, 1)]
        obs, reward, done, info = self.env.step(act)
        assert not done, info["exception"]
        assert (obs._shunt_bus[[0, 1]] == 1).all()
        assert (obs._shunt_bus[2:] == -1).all()

        # disconnect one of them and connect another one in the same action
        act = self.env.action_space()
        act.shunt_bus = [(0, -1), (2, 1)]
        obs, reward, done, info = self.env.step(act)
        assert not done, info["exception"]
        assert obs._shunt_bus[0] == -1
        assert obs._shunt_bus[1] == 1
        assert obs._shunt_bus[2] == 1
        assert (obs._shunt_bus[3:] == -1).all()


class TestShuntGymCompat(unittest.TestCase):
    """test the shunts can be used in the "Box" gymnasium spaces"""

    def setUp(self) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make("l2rpn_case14_sandbox",
                                    test=True,
                                    action_class=ShuntAction,
                                    _add_to_name=type(self).__name__)

    def tearDown(self) -> None:
        self.env.close()

    def test_act_space(self):
        act_space = BoxGymActSpace(self.env.action_space,
                                   attr_to_keep=["shunt_q"])
        assert act_space.shape == (type(self.env).n_shunt,)
        act = act_space.from_gym(np.array([-25.0], dtype=np.float32))
        assert act._modif_shunt
        assert act.shunt_q[0] == -25.0

    def test_gym_env_default_act_space(self):
        """the default (Dict) action space of GymEnv exposes the shunts"""
        gym_env = GymEnv(self.env)
        try:
            act_space = gym_env.action_space
            assert set(act_space.spaces.keys()) == {"shunt_p", "shunt_q", "shunt_bus"}
            n_shunt = type(self.env).n_shunt
            assert act_space["shunt_bus"].shape == (n_shunt,)
            assert (act_space["shunt_bus"].low == -1).all()
            assert (act_space["shunt_bus"].high == type(self.env).n_busbar_per_sub).all()

            gym_env.reset(seed=0)
            act_space.seed(0)
            gym_act = act_space.sample()
            act = act_space.from_gym(gym_act)
            assert act._modif_shunt
            *_, info = gym_env.step(gym_act)
            assert not info["is_ambiguous"]
            assert not info["is_illegal"]
        finally:
            gym_env.close()

    def test_act_space_bus(self):
        act_space = BoxGymActSpace(self.env.action_space,
                                   attr_to_keep=["shunt_bus"])
        assert act_space.shape == (type(self.env).n_shunt,)
        act = act_space.from_gym(np.array([-1], dtype=np.int32))
        assert act.shunt_bus[0] == -1

    def test_obs_space(self):
        obs_space = BoxGymObsSpace(self.env.observation_space,
                                   attr_to_keep=["_shunt_p", "_shunt_q", "_shunt_v", "_shunt_bus"])
        assert obs_space.shape == (4 * type(self.env).n_shunt,)
        obs = self.env.reset()
        gym_obs = obs_space.to_gym(obs)
        for attr_nm in ["_shunt_p", "_shunt_q", "_shunt_v", "_shunt_bus"]:
            beg_, end_ = obs_space.get_indexes(attr_nm)
            assert np.allclose(gym_obs[beg_:end_], getattr(obs, attr_nm)), f"error for {attr_nm}"

    def test_shunt_in_gym_loop(self):
        """the setpoint sent through the gymnasium space is applied on the grid"""
        act_space = BoxGymActSpace(self.env.action_space, attr_to_keep=["shunt_q"])
        self.env.set_id(0)
        self.env.reset()
        obs, reward, done, info = self.env.step(
            act_space.from_gym(np.array([-25.0], dtype=np.float32))
        )
        assert not info["is_illegal"]
        assert not info["is_ambiguous"]
        assert obs._shunt_q[0] < -25.0


class TestShuntNoShuntData(unittest.TestCase):
    """test the shunt api with a backend that does not support the shunts"""

    def setUp(self) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make("l2rpn_case14_sandbox",
                                    test=True,
                                    backend=EducPandaPowerBackend(),
                                    action_class=CompleteAction,
                                    _add_to_name=type(self).__name__)
        assert not type(self.env).shunts_data_available

    def tearDown(self) -> None:
        self.env.close()

    def test_getters(self):
        """the getters return empty vectors, the setters raise"""
        act = self.env.action_space()
        assert act.shunt_p.shape == (0,)
        assert act.shunt_p.dtype == dt_float
        assert act.shunt_q.shape == (0,)
        assert act.shunt_q.dtype == dt_float
        assert act.shunt_bus.shape == (0,)
        assert act.shunt_bus.dtype == dt_int
        with self.assertRaises(IllegalAction):
            act.shunt_q = [(0, -25.0)]

    def test_act_space(self):
        """the shunt part of the gymnasium space is empty and from_gym still works"""
        act_space = BoxGymActSpace(self.env.action_space,
                                   attr_to_keep=["redispatch", "shunt_q"])
        act_space_ref = BoxGymActSpace(self.env.action_space,
                                       attr_to_keep=["redispatch"])
        assert act_space.shape == act_space_ref.shape
        act = act_space.from_gym(np.zeros(act_space.shape, dtype=np.float32))
        assert not act._modif_shunt

    def test_obs_space(self):
        """the shunt part of the gymnasium space is empty and to_gym still works"""
        obs_space = BoxGymObsSpace(self.env.observation_space,
                                   attr_to_keep=["rho", "_shunt_q"])
        obs_space_ref = BoxGymObsSpace(self.env.observation_space,
                                       attr_to_keep=["rho"])
        assert obs_space.shape == obs_space_ref.shape
        obs = self.env.reset()
        gym_obs = obs_space.to_gym(obs)
        assert np.allclose(gym_obs, obs_space_ref.to_gym(obs))


if __name__ == "__main__":
    unittest.main()
