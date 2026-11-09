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
from grid2op.Action import CompleteAction
from grid2op.dtypes import dt_float
from grid2op.Environment.dispatch import LoadInjectionModule, RedispatchState, dispatch_contributions
from grid2op.Exceptions import ImpossibleRedispatching


class TestIssue777LoadInjectionDispatch(unittest.TestCase):
    """The change of load made by an injection of the action has to be compensated by the dispatch"""

    LOAD_ID = 1

    def setUp(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make(
                "educ_case14_redisp",
                test=True,
                action_class=CompleteAction,
                _add_to_name=type(self).__name__,
            )
        self.cls = type(self.env)
        self.load_ts = self._load_of_time_series()

    def tearDown(self):
        self.env.close()

    def _reset(self):
        return self.env.reset(seed=0, options={"time serie id": 0})

    def _load_of_time_series(self):
        """consumption of the load the first step, without any injection"""
        self._reset()
        obs, _, done, info = self.env.step(self.env.action_space({}))
        assert not done, info["exception"]
        return obs.load_p.copy()

    def _inject(self, delta):
        """adds `delta` MW to the load, the other loads are not modified (nan)"""
        self._reset()
        load_p = np.full(self.cls.n_load, fill_value=np.nan, dtype=dt_float)
        load_p[self.LOAD_ID] = self.load_ts[self.LOAD_ID] + delta
        return self.env.step(self.env.action_space({"injection": {"load_p": load_p}}))

    def test_no_change_of_load_needs_no_dispatch(self):
        obs, _, done, info = self._inject(0.0)
        assert not done, info["exception"]
        assert abs(obs.actual_dispatch.sum()) <= 1e-3

    def test_load_increase_is_compensated_by_the_generators(self):
        delta = 6.0
        obs, _, done, info = self._inject(delta)
        assert not done, info["exception"]
        # before the fix `actual_dispatch` was 0. and the slack generator took everything
        assert abs(obs.actual_dispatch.sum() - delta) <= 1e-2
        assert abs(obs.load_p[self.LOAD_ID] - self.load_ts[self.LOAD_ID] - delta) <= 1e-2

    def test_load_decrease_is_compensated_by_the_generators(self):
        delta = -6.0
        obs, _, done, info = self._inject(delta)
        assert not done, info["exception"]
        assert abs(obs.actual_dispatch.sum() - delta) <= 1e-2

    def test_generators_respect_their_ramps(self):
        _ = self._inject(0.0)
        obs_ref = self.env.get_obs()
        obs, _, done, info = self._inject(6.0)
        assert not done, info["exception"]
        assert (obs.gen_p - obs_ref.gen_p <= self.cls.gen_max_ramp_up + 1e-2).all()
        assert (obs.gen_p <= self.cls.gen_pmax + 1e-2).all()

    def test_too_large_increase_ends_the_episode(self):
        # more than what all the generators can follow in one step (the reproduction of the issue)
        delta = 2.0 * self.cls.gen_max_ramp_up.sum()
        obs, _, done, info = self._inject(delta)
        assert done
        assert any(isinstance(exc, ImpossibleRedispatching) for exc in info["exception"])

    def test_dispatch_goes_back_to_zero_when_the_injection_stops(self):
        _ = self._inject(6.0)
        obs, _, done, info = self.env.step(self.env.action_space({}))
        assert not done, info["exception"]
        assert abs(obs.actual_dispatch.sum()) <= 1e-2

    def test_not_modified_loads_do_not_count(self):
        # an injection that sets the load to the value of the time series changes nothing
        self._reset()
        load_p = np.full(self.cls.n_load, fill_value=np.nan, dtype=dt_float)
        load_p[self.LOAD_ID] = self.load_ts[self.LOAD_ID]
        obs, _, done, info = self.env.step(self.env.action_space({"injection": {"load_p": load_p}}))
        assert not done, info["exception"]
        assert abs(obs.actual_dispatch.sum()) <= 1e-3

    def test_without_injection_the_dispatch_is_unchanged(self):
        self._reset()
        for _ in range(3):
            obs, _, done, info = self.env.step(self.env.action_space({}))
            assert not done, info["exception"]
            assert abs(obs.actual_dispatch.sum()) <= 1e-3


class TestIssue777Contribution(unittest.TestCase):
    def test_load_is_one_of_the_contributions(self):
        state = RedispatchState.allocate(n_gen=3, n_storage=0, n_load=2)
        assert dispatch_contributions(state)["load"] == 0.0
        state.load_injection_mw = dt_float(4.5)
        contributions = dispatch_contributions(state)
        # the loads consume more: the generators have to produce more
        assert contributions["load"] == 4.5
        assert contributions["storage"] == 0.0


class _FakeAction:
    def __init__(self, load_p=None):
        self._dict_inj = {} if load_p is None else {"load_p": np.array(load_p, dtype=dt_float)}


class _FakeBackendAction:
    def __init__(self, load_detached):
        self._load_detached = np.array(load_detached, dtype=bool)

    def get_load_detached(self):
        return self._load_detached


class TestIssue777LoadInjectionModule(unittest.TestCase):
    def setUp(self):
        self.state = RedispatchState.allocate(n_gen=2, n_storage=0, n_load=3)
        self.module = LoadInjectionModule(None)
        self.no_detached = _FakeBackendAction([False, False, False])
        self.ts_load = np.array([10.0, 20.0, 30.0], dtype=dt_float)
        self.all_set = np.ones(3, dtype=bool)
        self.none_set = np.zeros(3, dtype=bool)

    def _compute(self, action, load_set_by_ts, backend_action=None, new_load_p=None):
        return self.module.compute(
            action,
            self.ts_load if new_load_p is None else new_load_p,
            load_set_by_ts,
            self.state,
            self.no_detached if backend_action is None else backend_action,
        )

    def test_no_injection_is_nothing_to_compensate(self):
        assert self._compute(_FakeAction(), self.all_set) == 0.0
        assert self.state.load_injection_mw == 0.0

    def test_only_finite_entries_count(self):
        res = self._compute(_FakeAction([np.nan, 25.0, np.nan]), self.all_set)
        assert res == 5.0
        assert self.state.load_injection_mw == 5.0

    def test_injection_stops_when_the_time_series_sets_the_load_again(self):
        _ = self._compute(_FakeAction([np.nan, 25.0, np.nan]), self.all_set)
        res = self._compute(_FakeAction(), self.all_set)
        # the load is back to the time series: the generators have to come back too
        assert res == -5.0
        assert self.state.load_injection_excess.sum() == 0.0

    def test_injection_lasts_when_the_time_series_does_not_set_the_load(self):
        _ = self._compute(_FakeAction([np.nan, 25.0, np.nan]), self.none_set)
        # next step: nothing in the time series, the load keeps its (injected) value
        new_load_p = np.array([10.0, 25.0, 30.0], dtype=dt_float)
        res = self._compute(_FakeAction(), self.none_set, new_load_p=new_load_p)
        assert res == 0.0
        assert self.state.load_injection_excess.sum() == 5.0
        # setting the same value again changes nothing either
        res = self._compute(_FakeAction([np.nan, 25.0, np.nan]), self.none_set, new_load_p=new_load_p)
        assert res == 0.0

    def test_detached_load_is_ignored(self):
        res = self._compute(
            _FakeAction([np.nan, 25.0, 40.0]), self.all_set, _FakeBackendAction([False, False, True])
        )
        assert res == 5.0


if __name__ == "__main__":
    unittest.main()
