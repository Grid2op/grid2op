# Copyright (c) 2026, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import json
import os
import pickle
import shutil
import tempfile
import unittest
import warnings

import numpy as np

import grid2op
from grid2op.Agent import DoNothingAgent
from grid2op.Environment import MaskedEnvironment
from grid2op.Environment.protection import (Protection,
                                            ProtectionConfig,
                                            ProtectionState,
                                            legacy_from_parameters,
                                            PROTECTIONS_FILE_NAME)
from grid2op.Environment.protection.protection_solver import compute_engaged, cascade_iteration
from grid2op.Exceptions import EnvError
from grid2op.Parameters import Parameters
from grid2op.Runner import Runner


class TestProtectionConfig(unittest.TestCase):
    """pure numpy tests, no environment"""
    def test_protection_checks(self):
        with self.assertRaises(EnvError):
            Protection(line_id=0, side="middle", limit=1., delay=0)
        with self.assertRaises(EnvError):
            Protection(line_id=0, side="or", limit=1., delay=-1)
        with self.assertRaises(EnvError):
            Protection(line_id=0, side="or", limit=0., delay=0)
        with self.assertRaises(EnvError):
            Protection(line_id=-1, side="or", limit=1., delay=0)

    def test_soa_layout(self):
        prots = [Protection(1, "or", 150., 0, name="a"),
                 Protection(1, "ex", 125., 3, name="b"),
                 Protection(0, "or", 100., 2, in_service=False)]
        cfg = ProtectionConfig.from_protections(prots, n_line=2)
        assert cfg.n_prot == 3
        assert np.array_equal(cfg.line_id, [1, 1, 0])
        assert np.array_equal(cfg.side_is_ex, [False, True, False])
        assert np.array_equal(cfg.side, ["or", "ex", "or"])
        assert np.allclose(cfg.limit, [150., 125., 100.])
        assert np.array_equal(cfg.delay, [0, 3, 2])
        assert np.array_equal(cfg.in_service, [True, True, False])
        assert cfg.has_ex_side
        assert cfg.to_protections() == prots
        # structure is frozen, operational status is not
        with self.assertRaises(ValueError):
            cfg.limit[0] = 2.
        with self.assertRaises(ValueError):
            cfg.line_id[0] = 0
        cfg.in_service[2] = True
        # copies do not share the operational status
        cfg2 = cfg.copy()
        cfg2.in_service[0] = False
        assert cfg.in_service[0]
        assert cfg.same_structure(cfg2)
        assert cfg != cfg2
        # pickle (multiprocessing) keeps everything, including the read only flags
        cfg3 = pickle.loads(pickle.dumps(cfg))
        assert cfg3 == cfg
        with self.assertRaises(ValueError):
            cfg3.delay[0] = 2
        cfg3.in_service[0] = False

    def test_errors(self):
        with self.assertRaises(EnvError):
            # line id too high
            ProtectionConfig.from_protections([Protection(2, "or", 1., 0)], n_line=2)
        with self.assertRaises(EnvError):
            # duplicated names
            ProtectionConfig.from_protections([Protection(0, "or", 1., 0, name="a"),
                                               Protection(1, "or", 1., 0, name="a")])
        with self.assertRaises(EnvError):
            Protection.from_dict({"line_id": 0, "limit": 1., "unknown_key": 1})
        with self.assertRaises(EnvError):
            Protection.from_dict({"line_name": "l_x", "limit": 1.}, name_line=["l_0", "l_1"])

    def test_consistency(self):
        P = Protection
        # valid: limits strictly increasing and delays strictly decreasing on each side
        ProtectionConfig.from_protections([P(0, "or", 100., 4), P(0, "or", 120., 1), P(0, "or", 200., 0),
                                           # other side / other line: independent
                                           P(0, "ex", 100., 4), P(1, "or", 100., 4)])
        invalid = {
            "same limit": [P(0, "or", 100., 4), P(0, "or", 100., 1)],
            "same delay": [P(0, "or", 100., 4), P(0, "or", 120., 4)],
            "higher limit, higher delay": [P(0, "or", 100., 1), P(0, "or", 120., 4)],
            "same limit, same delay": [P(0, "ex", 100., 1), P(0, "ex", 100., 1)],
            "three stages, one wrong": [P(0, "or", 100., 4), P(0, "or", 120., 1), P(0, "or", 110., 0)],
        }
        for case, prots in invalid.items():
            with self.assertRaises(EnvError, msg=case):
                ProtectionConfig.from_protections(prots)
        # also checked for protections out of service
        with self.assertRaises(EnvError):
            ProtectionConfig.from_protections([P(0, "or", 100., 4), P(0, "or", 100., 1, in_service=False)])

    def test_reference(self):
        P = Protection
        cfg = ProtectionConfig.from_protections([P(0, "or", 200., 0), P(0, "or", 100., 4), P(0, "or", 120., 1),
                                                 P(0, "ex", 50., 0, in_service=False),
                                                 P(2, "ex", 30., 2)])
        ref_or, ref_ex = cfg.reference_limits(3)
        assert np.allclose(ref_or, [100., np.inf, np.inf])
        assert np.allclose(ref_ex, [50., np.inf, 30.])
        assert np.array_equal(cfg.is_reference(), [False, True, False, True, True])

    def test_from_dict_and_json(self):
        content = {"protections": [{"line_name": "l_1", "side": "ex", "limit": 120., "delay": 2, "name": "p"},
                                   {"line_id": 0, "limit": 200.}]}
        cfg = ProtectionConfig.from_raw(content, n_line=2, name_line=["l_0", "l_1"])
        assert np.array_equal(cfg.line_id, [1, 0])
        assert np.array_equal(cfg.side, ["ex", "or"])
        assert np.array_equal(cfg.delay, [2, 0])
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "prot.json")
            cfg.to_json(path)
            cfg2 = ProtectionConfig.from_json(path, n_line=2)
        assert cfg2 == cfg

    def test_get_ids(self):
        cfg = legacy_from_parameters(Parameters(), np.array([100., 200., 300.]))
        assert np.array_equal(cfg.get_ids("l1_soft"), [3])
        assert np.array_equal(cfg.get_ids([0, "l2_hard"]), [0, 4])
        assert np.array_equal(cfg.get_ids(cfg.line_id == 1), [2, 3])
        with self.assertRaises(EnvError):
            cfg.get_ids("unknown")
        with self.assertRaises(EnvError):
            cfg.get_ids(6)

    def test_legacy_from_parameters(self):
        th = np.array([100., 200., 400.])
        params = Parameters()
        params.HARD_OVERFLOW_THRESHOLD = 3.
        params.PROTECTION_THRESHOLD = 1.25
        params.NB_TIMESTEP_OVERFLOW_ALLOWED = 4
        cfg = legacy_from_parameters(params, th)
        assert cfg.n_prot == 6
        assert np.array_equal(cfg.line_id, [0, 0, 1, 1, 2, 2])
        assert not cfg.has_ex_side
        # "hard": acts above HARD * th, "soft" (the reference): limit th
        assert np.allclose(cfg.limit[0::2], 3. / 1.25 * th)
        assert np.allclose(cfg.limit[1::2], th)
        assert np.allclose(params.PROTECTION_THRESHOLD * cfg.limit[0::2], 3. * th)
        assert np.array_equal(cfg.delay, [0, 4] * 3)
        assert np.array_equal(cfg.is_reference(), [False, True] * 3)
        assert np.allclose(cfg.reference_limits(3)[0], th)
        assert cfg.name[0] == "l0_hard"
        assert cfg.name[1] == "l0_soft"
        # the in_service status can be kept
        in_service = np.array([True, False] * 3)
        cfg = legacy_from_parameters(params, th, in_service=in_service)
        assert np.array_equal(cfg.in_service, in_service)
        # without delay: only the reference protection (instantaneous)
        params.NB_TIMESTEP_OVERFLOW_ALLOWED = 0
        cfg = legacy_from_parameters(params, th)
        assert cfg.n_prot == 3
        assert np.allclose(cfg.limit, th)
        assert np.array_equal(cfg.delay, [0, 0, 0])
        assert np.array_equal(cfg.name, ["l0_soft", "l1_soft", "l2_soft"])

    def test_state_in_service(self):
        cfg = ProtectionConfig.from_protections([Protection(0, "or", 1., 5), Protection(1, "or", 1., 5)])
        state = ProtectionState.from_config(cfg)
        engaged = np.array([True, True])
        state.update(cfg, engaged)
        state.update(cfg, engaged)
        assert np.array_equal(state.counter, [2, 2])
        # out of service: frozen
        cfg.in_service[1] = False
        state.update(cfg, engaged)
        assert np.array_equal(state.counter, [3, 2])
        state.update(cfg, ~engaged)
        assert np.array_equal(state.counter, [0, 2])
        # back in service: restarts from 0
        cfg.in_service[1] = True
        state.update(cfg, engaged)
        assert np.array_equal(state.counter, [1, 1])
        assert np.array_equal(state.line_counter(cfg, 3), [1, 1, 0])

    def test_solver(self):
        cfg = ProtectionConfig.from_protections([Protection(0, "or", 20., 0),
                                                 Protection(0, "or", 10., 2),
                                                 Protection(1, "ex", 10., 0),
                                                 Protection(1, "or", 10., 0, in_service=False)])
        a_or = np.array([15., 15.])
        a_ex = np.array([15., 5.])
        status = np.array([True, True])
        engaged = compute_engaged(cfg, a_or, a_ex, 1., status)
        assert np.array_equal(engaged, [False, True, False, True])
        # the protection threshold multiplies all the limits
        assert np.array_equal(compute_engaged(cfg, a_or, a_ex, 0.7, status), [True, True, False, True])
        assert np.array_equal(compute_engaged(cfg, a_or, a_ex, 1.6, status), [False, False, False, False])
        counter = np.array([0, 2, 0, 0])
        increased = np.zeros(4, dtype=bool)
        tripped = cascade_iteration(cfg, counter, increased, engaged, 2)
        # delayed protection of line 0 trips (3 > 2), "or" protection of line 1 is out of service
        assert np.array_equal(tripped, [True, False])
        assert np.array_equal(counter, [0, 3, 0, 0])
        # at most one increment per step
        tripped = cascade_iteration(cfg, counter, increased, engaged, 2)
        assert np.array_equal(counter, [0, 3, 0, 0])
        # global switch: counters are updated, nothing trips
        counter = np.array([0, 2, 0, 0])
        increased[:] = False
        tripped = cascade_iteration(cfg, counter, increased, engaged, 2, protections_disabled=True)
        assert not tripped.any()
        assert np.array_equal(counter, [0, 3, 0, 0])


class _BaseProtectionEnv:
    """line 1 is overloaded (between 1.15 and 1.35 x its thermal limit, 110 A) for the first steps of the
    scenario, all the other lines are far from their limits (see `setUp`)"""
    line_id = 1
    th_line = 110.

    def setUp(self) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make("l2rpn_case14_sandbox", test=True, _add_to_name=type(self).__name__)
        obs = self.env.reset(seed=0, options={"time serie id": 0})
        self.th_lim = 10. * np.maximum(obs.a_or, obs.a_ex)
        self.th_lim[self.line_id] = self.th_line
        return super().setUp()

    def tearDown(self) -> None:
        self.env.close()
        return super().tearDown()

    def _reset(self):
        return self.env.reset(seed=0, options={"time serie id": 0, "thermal limit": self.th_lim})

    def _step(self):
        obs, reward, done, info = self.env.step(self.env.action_space())
        assert not done
        return obs, info

    def _prot(self, rel_limit, delay, **kwargs):
        """protection on the "or" side of self.line_id, limit relative to its thermal limit"""
        return Protection(self.line_id, "or", rel_limit * self.th_line, delay, **kwargs)


class TestProtectionEnv(_BaseProtectionEnv, unittest.TestCase):
    def test_default_config(self):
        obs = self._reset()
        cfg = self.env.get_protection_config()
        assert cfg == legacy_from_parameters(self.env.parameters, self.th_lim)
        assert not self.env._protection_is_custom
        assert obs.protection_counters.shape == (2 * type(self.env).n_line, )
        assert np.array_equal(obs.protection_line_id, cfg.line_id)
        assert np.array_equal(obs.protection_side, cfg.side)
        assert self.env.get_params_for_runner()["protections"] is None
        # the thermal limit is the limit of the reference protection
        assert np.allclose(self.env.get_thermal_limit(), self.th_lim)

    def test_default_follows_parameters_and_thermal_limits(self):
        params = self.env.parameters
        params.HARD_OVERFLOW_THRESHOLD = 3.
        params.PROTECTION_THRESHOLD = 1.25
        params.NB_TIMESTEP_OVERFLOW_ALLOWED = 5
        self.env.change_parameters(params)
        self._reset()
        cfg = self.env.get_protection_config()
        assert np.allclose(cfg.limit[0::2], 3. / 1.25 * self.th_lim)
        assert np.allclose(cfg.limit[1::2], self.th_lim)
        assert np.all(cfg.delay[1::2] == 5)
        # thermal limits
        th_lim = 2. * self.th_lim
        self.env.set_thermal_limit(th_lim)
        assert np.allclose(self.env.get_protection_config().limit[1::2], th_lim)

        # custom configuration does not follow the parameters nor the thermal limits
        self.env.set_protections([Protection(0, "or", 150., 1)])
        params.HARD_OVERFLOW_THRESHOLD = 4.
        self.env.change_parameters(params)
        self._reset()
        cfg = self.env.get_protection_config()
        assert cfg.n_prot == 1
        assert np.allclose(cfg.limit, 150.)
        with self.assertWarns(UserWarning):
            self.env.set_thermal_limit(self.th_lim)
        assert np.allclose(self.env.get_protection_config().limit, 150.)

        # back to the legacy ones
        self.env.set_protections(None)
        assert np.allclose(self.env.get_protection_config().limit[0::2], 4. / 1.25 * self.th_lim)

    def test_default_identical_to_explicit(self):
        """an environment with an explicit config equal to the legacy one behaves exactly the same"""
        th_lim = 1. * self.th_lim
        th_lim[[1, 4, 9]] = [120., 100., 600.]  # a few overloads, with cascades
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env2 = grid2op.make("l2rpn_case14_sandbox", test=True, _add_to_name=type(self).__name__)
        try:
            env2.set_protections(legacy_from_parameters(env2.parameters, th_lim))
            assert env2._protection_is_custom
            options = {"time serie id": 0, "thermal limit": th_lim}
            obs1 = self.env.reset(seed=0, options=options)
            obs2 = env2.reset(seed=0, options=options)
            for _ in range(10):
                obs1, r1, done1, info1 = self.env.step(self.env.action_space())
                obs2, r2, done2, info2 = env2.step(env2.action_space())
                assert done1 == done2
                assert np.array_equal(obs1.line_status, obs2.line_status)
                assert np.array_equal(info1["disc_lines"], info2["disc_lines"])
                assert np.array_equal(obs1.timestep_protection_engaged, obs2.timestep_protection_engaged)
                assert np.array_equal(obs1.protection_counters, obs2.protection_counters)
                assert np.allclose(obs1.a_or, obs2.a_or)
                if done1:
                    break
            # there has been some disconnections
            assert (~obs1.line_status).any()
        finally:
            env2.close()

    def test_ex_side_only(self):
        # line 15 is a transformer: the current is much higher on its "ex" side
        line_id = 15
        obs = self._reset()
        limit = 0.5 * (obs.a_or[line_id] + obs.a_ex[line_id])
        assert obs.a_ex[line_id] > 1.5 * limit
        assert obs.a_or[line_id] < 0.5 * limit

        # protection on the "or" side: no trip
        self.env.set_protections([Protection(line_id, "or", limit, 0)])
        obs, info = self._step()
        assert obs.line_status[line_id]
        assert obs.protection_counters[0] == 0

        # protection on the "ex" side: trip
        self.env.set_protections([Protection(line_id, "ex", limit, 0)])
        obs, info = self._step()
        assert not obs.line_status[line_id]
        assert info["disc_lines"][line_id] == 0

    def test_two_stages(self):
        self.env.set_protections([self._prot(1.1, 3, name="slow"),
                                  self._prot(1.15, 1, name="fast"),
                                  self._prot(5.0, 0, name="inst")])
        obs = self._reset()
        assert (obs.protection_counters == 0).all()
        obs, info = self._step()
        assert obs.a_or[self.line_id] > 1.15 * self.th_line
        assert obs.line_status[self.line_id]
        assert np.array_equal(obs.protection_counters, [1, 1, 0])
        assert obs.timestep_protection_engaged[self.line_id] == 1
        obs, info = self._step()
        # "fast" stage: 2 > 1
        assert not obs.line_status[self.line_id]
        assert info["disc_lines"][self.line_id] == 0

        # without the fast stage, the slow one trips after 4 steps
        self.env.set_protection_in_service("fast", False)
        obs = self._reset()
        for ts in range(1, 4):
            obs, info = self._step()
            assert obs.line_status[self.line_id], f"error for {ts}"
            assert np.array_equal(obs.protection_counters, [ts, 0, 0]), f"error for {ts}"
        obs, info = self._step()
        assert not obs.line_status[self.line_id]

    def test_protection_threshold(self):
        """PROTECTION_THRESHOLD multiplies the limit of every protection, instantaneous ones included"""
        self.env.set_protections([self._prot(1.5, 0, name="inst")])
        obs = self._reset()
        obs, info = self._step()
        assert obs.line_status[self.line_id]  # current < 1.5 * th
        params = self.env.parameters
        params.PROTECTION_THRESHOLD = 0.75  # now engaged above 1.125 * th
        self.env.change_parameters(params)
        obs = self._reset()
        obs, info = self._step()
        assert not obs.line_status[self.line_id]

    def test_in_service(self):
        self.env.set_protections([self._prot(1.1, 1, in_service=False, name="p")])
        obs = self._reset()
        for _ in range(3):
            obs, info = self._step()
            assert obs.line_status[self.line_id]
            assert obs.protection_counters[0] == 0
            # no protection in service on this line
            assert obs.timestep_protection_engaged[self.line_id] == 0

        self.env.set_protection_in_service("p", True)
        obs, info = self._step()
        assert obs.line_status[self.line_id]
        assert obs.protection_counters[0] == 1
        # out of service: frozen
        self.env.set_protection_in_service(0, False)
        for _ in range(2):
            obs, info = self._step()
            assert obs.line_status[self.line_id]
            assert obs.protection_counters[0] == 1
        # back in service: restarts from 0
        self.env.set_protection_in_service([0], True)
        assert self.env.get_protection_counters()[0] == 0
        obs, info = self._step()
        assert obs.line_status[self.line_id]
        assert obs.protection_counters[0] == 1
        obs, info = self._step()
        assert not obs.line_status[self.line_id]

    def test_in_service_default_config(self):
        # protections of the default config can be put out of service, it survives a reset
        cfg = self.env.get_protection_config()
        self.env.set_protection_in_service(cfg.line_id == self.line_id, False)
        obs = self._reset()
        assert not self.env._protection_is_custom
        for _ in range(5):
            obs, info = self._step()
            assert obs.line_status[self.line_id]
        self.env.set_protection_in_service(cfg.line_id == self.line_id, True)
        for _ in range(2):
            obs, info = self._step()
            assert obs.line_status[self.line_id]
        obs, info = self._step()
        assert not obs.line_status[self.line_id]

    def test_no_overflow_disconnection(self):
        params = self.env.parameters
        params.NO_OVERFLOW_DISCONNECTION = True
        self.env.change_parameters(params)
        self.env.set_protections([self._prot(1.1, 1), self._prot(1.12, 0)])
        obs = self._reset()
        for ts in range(1, 5):
            obs, info = self._step()
            assert obs.line_status[self.line_id]
            # counters are still updated
            assert np.array_equal(obs.protection_counters, [ts, ts])
            assert obs.timestep_protection_engaged[self.line_id] == ts

    def test_add_protection(self):
        n_prot = 2 * type(self.env).n_line
        obs = self._reset()
        obs, info = self._step()
        counters = obs.protection_counters.copy()
        assert counters[2 * self.line_id + 1] == 1
        new_id = self.env.add_protection(type(self.env).name_line[self.line_id], "or",
                                         1.125 * self.th_line, 1, name="extra")
        assert new_id == n_prot
        assert self.env._protection_is_custom
        # counters of the existing protections are kept
        assert np.array_equal(self.env.get_protection_counters()[:n_prot], counters)
        obs, info = self._step()
        assert obs.protection_counters[new_id] == 1
        assert obs.line_status[self.line_id]
        obs, info = self._step()
        # the new protection trips first (the default one needs 3 steps)
        assert not obs.line_status[self.line_id]
        assert self.env.get_protections()[new_id] == Protection(self.line_id, "or", 1.125 * self.th_line, 1,
                                                                name="extra")
        # inconsistent with the existing ones (same delay as the reference protection)
        with self.assertRaises(EnvError):
            self.env.add_protection(self.line_id, "or", 1.5 * self.th_line, 2)

    def test_copy(self):
        cfg = [self._prot(1.1, 2, name="p"),
               Protection(15, "ex", 3. * self.th_lim[15], 0, name="q")]
        self.env.set_protections(cfg)
        obs = self._reset()
        obs, info = self._step()
        env_cpy = self.env.copy()
        try:
            assert env_cpy.get_protection_config() == self.env.get_protection_config()
            assert np.array_equal(env_cpy.get_protection_counters(), [1, 0])
            # independent from each other
            assert self.env._protection_config.in_service is not env_cpy._protection_config.in_service
            assert self.env._protection_state.counter is not env_cpy._protection_state.counter
            for _ in range(2):
                obs, *_ = self.env.step(self.env.action_space())
                obs_cpy, *_ = env_cpy.step(env_cpy.action_space())
                assert np.array_equal(obs.protection_counters, obs_cpy.protection_counters)
                assert np.array_equal(obs.line_status, obs_cpy.line_status)
            assert not obs_cpy.line_status[self.line_id]
        finally:
            env_cpy.close()

    def test_simulate(self):
        self.env.set_protections([self._prot(1.1, 1, name="p")])
        obs = self._reset()
        sim_obs, *_ = obs.simulate(self.env.action_space())
        assert sim_obs.line_status[self.line_id]
        assert sim_obs.protection_counters[0] == 1
        obs, info = self._step()
        assert obs.protection_counters[0] == 1
        # counter is taken from the observation, simulated line trips
        sim_obs, *_ = obs.simulate(self.env.action_space())
        assert not sim_obs.line_status[self.line_id]

        # out of service also applies to simulate
        self.env.set_protection_in_service("p", False)
        sim_obs, *_ = obs.simulate(self.env.action_space())
        assert sim_obs.line_status[self.line_id]
        self.env.set_protection_in_service("p", True)

        # and to the forecast env
        for_env = obs.get_forecast_env()
        try:
            assert for_env.get_protection_config() == self.env.get_protection_config()
            for_obs = for_env.reset()
            assert for_obs.protection_counters[0] == 1
            for_obs, *_ = for_env.step(for_env.action_space())
            assert not for_obs.line_status[self.line_id]
        finally:
            for_env.close()

        # back to the default protections: simulate does not trip (NB_TIMESTEP_OVERFLOW_ALLOWED=2)
        self.env.set_protections(None)
        obs = self._reset()
        obs, info = self._step()
        sim_obs, *_ = obs.simulate(self.env.action_space())
        assert sim_obs.line_status[self.line_id]

    def test_runner(self):
        cfg = ProtectionConfig.from_protections([self._prot(1.1, 1)])
        self.env.set_protections(cfg)
        params = self.env.get_params_for_runner()
        assert params["protections"] == cfg
        runner = Runner(**params, agentClass=DoNothingAgent)
        env_runner = runner.init_env()
        try:
            assert env_runner.get_protection_config() == cfg
        finally:
            env_runner.close()

    def test_no_protection_at_reset(self):
        """the observation given by reset is the initial state of the grid: no protection acts
        on it, whatever its delay (even instantaneous ones), only a real step can trip a line"""
        th_lim = 1. * self.th_lim
        th_lim[self.line_id] = 1.  # flow on this line is far above any limit
        for delay in [0, 1, 3]:
            self.env.set_protections([Protection(self.line_id, "or", 1.5, delay, name="p"),
                                      Protection(self.line_id, "ex", 1.5, delay, name="q")])
            obs = self.env.reset(seed=0, options={"time serie id": 0, "thermal limit": th_lim})
            assert obs.line_status[self.line_id], f"error for delay {delay}"
            assert obs.a_or[self.line_id] > 50., f"error for delay {delay}"
            assert (obs.protection_counters == 0).all(), f"error for delay {delay}"
            assert obs.timestep_protection_engaged[self.line_id] == 0, f"error for delay {delay}"
            for ts in range(1, delay + 1):
                obs, info = self._step()
                assert obs.line_status[self.line_id], f"error for delay {delay} at step {ts}"
                assert (obs.protection_counters == ts).all(), f"error for delay {delay} at step {ts}"
            obs, info = self._step()
            assert not obs.line_status[self.line_id], f"error for delay {delay}"
            assert info["disc_lines"][self.line_id] == 0, f"error for delay {delay}"

        # same with the legacy protections: the "hard overflow" does not act at reset
        self.env.init_protection_legacy()
        obs = self.env.reset(seed=0, options={"time serie id": 0, "thermal limit": th_lim})
        assert obs.rho[self.line_id] > self.env.parameters.HARD_OVERFLOW_THRESHOLD
        assert obs.line_status[self.line_id]
        obs, info = self._step()
        assert not obs.line_status[self.line_id]

    def test_no_protection_at_reset_init_ts(self):
        """also when the first steps of the time series are skipped at reset"""
        th_lim = 1. * self.th_lim
        th_lim[self.line_id] = 1.
        for init_ts in [2, 3]:
            obs = self.env.reset(seed=0, options={"time serie id": 0, "thermal limit": th_lim, "init ts": init_ts})
            assert obs.line_status[self.line_id], f"error for {init_ts}"
            assert (obs.protection_counters == 0).all(), f"error for {init_ts}"

    def test_init_protection_legacy(self):
        self._reset()
        self.env.set_protections([Protection(0, "or", 150., 1)])
        # back to the protections following the parameters of the env
        self.env.init_protection_legacy()
        assert not self.env._protection_is_custom
        assert self.env.get_protection_config() == legacy_from_parameters(self.env.parameters, self.th_lim)

        # fixed legacy protections from other parameters
        params = Parameters()
        params.HARD_OVERFLOW_THRESHOLD = 3.
        params.PROTECTION_THRESHOLD = 1.25
        params.NB_TIMESTEP_OVERFLOW_ALLOWED = 4
        self.env.set_protection_in_service(0, False)
        self.env.init_protection_legacy(params)
        assert self.env._protection_is_custom
        cfg = self.env.get_protection_config()
        assert cfg == legacy_from_parameters(params, self.th_lim)
        assert cfg.in_service.all()
        assert np.array_equal(cfg.name[:2], ["l0_hard", "l0_soft"])
        # they do not follow the parameters of the environment
        env_params = self.env.parameters
        env_params.NB_TIMESTEP_OVERFLOW_ALLOWED = 1
        self.env.change_parameters(env_params)
        self._reset()
        assert np.all(self.env.get_protection_config().delay[1::2] == 4)
        with self.assertRaises(EnvError):
            self.env.init_protection_legacy({"HARD_OVERFLOW_THRESHOLD": 3.})

    def test_legacy_no_delay(self):
        """NB_TIMESTEP_OVERFLOW_ALLOWED = 0: one (instantaneous) protection per line"""
        params = self.env.parameters
        params.NB_TIMESTEP_OVERFLOW_ALLOWED = 0
        self.env.change_parameters(params)
        obs = self._reset()
        assert self.env.get_protection_config().n_prot == type(self.env).n_line
        assert obs.protection_counters.shape == (type(self.env).n_line,)
        obs, info = self._step()
        assert not obs.line_status[self.line_id]
        # and back
        params.NB_TIMESTEP_OVERFLOW_ALLOWED = 2
        self.env.change_parameters(params)
        obs = self._reset()
        assert self.env.get_protection_config().n_prot == 2 * type(self.env).n_line

    def test_errors(self):
        with self.assertRaises(EnvError):
            self.env.set_protections([Protection(type(self.env).n_line, "or", 110., 1)])
        with self.assertRaises(EnvError):
            self.env.add_protection("unknown_line", "or", 110., 1)
        with self.assertRaises(EnvError):
            self.env.set_protection_in_service("unknown_protection", False)
        with self.assertRaises(EnvError):
            # inconsistent
            self.env.set_protections([self._prot(1.1, 1), self._prot(1.2, 1)])


class TestProtectionMasked(unittest.TestCase):
    def test_masked_default_config(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = grid2op.make("l2rpn_case14_sandbox", test=True, _add_to_name=type(self).__name__)
            lines_of_interest = np.zeros(type(env).n_line, dtype=bool)
            lines_of_interest[:3] = True
            env_masked = MaskedEnvironment(env, lines_of_interest=lines_of_interest)
        try:
            for init_legacy in [False, True]:
                if init_legacy:
                    # masked lines also with init_protection_legacy
                    env_masked.init_protection_legacy(env_masked.parameters)
                cfg = env_masked.get_protection_config()
                not_interest = ~lines_of_interest[cfg.line_id]
                is_ref = cfg.is_reference()
                th = env_masked._thermal_limit_a
                # reference protection: same limit (same rho), "infinite" delay
                assert np.allclose(cfg.limit[is_ref], th)
                assert (cfg.delay[not_interest & is_ref] == MaskedEnvironment.INF_VAL_TS_OVERFLOW_ALLOW).all()
                # instantaneous one: "infinite" limit
                assert (cfg.limit[not_interest & ~is_ref] >= MaskedEnvironment.INF_VAL_THM_LIM * 0.99).all()
                assert (cfg.delay[~not_interest] == np.array([0, 2] * 3)).all()
        finally:
            env_masked.close()
            env.close()


class TestProtectionFile(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.env_path = os.path.join(self.tmp_dir.name, "l2rpn_case14_sandbox")
        shutil.copytree(os.path.join(os.path.dirname(grid2op.__file__), "data", "l2rpn_case14_sandbox"),
                        self.env_path,
                        ignore=shutil.ignore_patterns("_grid2op_classes", "__pycache__"))
        return super().setUp()

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()
        return super().tearDown()

    def test_read_file(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = grid2op.make(self.env_path, test=True, _add_to_name=type(self).__name__ + "ref")
        name_line = type(env).name_line
        env.close()
        content = {"protections": [
            {"line_name": name_line[2], "side": "ex", "limit": 125., "delay": 3, "name": "a"},
            {"line_id": 0, "side": "or", "limit": 150., "delay": 0, "in_service": False}
        ]}
        with open(os.path.join(self.env_path, PROTECTIONS_FILE_NAME), "w", encoding="utf-8") as f:
            json.dump(content, f)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = grid2op.make(self.env_path, test=True, _add_to_name=type(self).__name__)
        try:
            assert env.get_protections() == [Protection(2, "ex", 125., 3, name="a"),
                                             Protection(0, "or", 150., 0, in_service=False)]
            obs = env.reset()
            assert obs.protection_counters.shape == (2,)
            # the protections are also used by the runner
            assert env.get_params_for_runner()["protections"] == env.get_protection_config()
            assert np.array_equal(obs.protection_side, ["ex", "or"])
        finally:
            env.close()

    def test_inconsistent_file(self):
        content = [{"line_id": 0, "limit": 125., "delay": 3}, {"line_id": 0, "limit": 150., "delay": 3}]
        with open(os.path.join(self.env_path, PROTECTIONS_FILE_NAME), "w", encoding="utf-8") as f:
            json.dump(content, f)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with self.assertRaises(EnvError):
                grid2op.make(self.env_path, test=True, _add_to_name=type(self).__name__ + "bad")


if __name__ == "__main__":
    unittest.main()
