# Copyright (c) 2026, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import tempfile
import unittest
import warnings

import numpy as np

import grid2op
from grid2op.Action import CompleteAction
from grid2op.Agent import BaseAgent, DoNothingAgent
from grid2op.dtypes import dt_float
from grid2op.Environment.dispatch import LoadInjectionModule, RedispatchState, dispatch_contributions
from grid2op.Episode import EpisodeData
from grid2op.Exceptions import ImpossibleRedispatching
from grid2op.Runner import Runner


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


def _make_env(name, name_suffix):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        return grid2op.make(name, test=True, action_class=CompleteAction, _add_to_name=name_suffix)


def _injection(env, load_id, value):
    """action that sets the active consumption of one load, the others are not modified (nan)"""
    load_p = np.full(type(env).n_load, fill_value=np.nan, dtype=dt_float)
    load_p[load_id] = value
    return env.action_space({"injection": {"load_p": load_p}})


def _exceptions(info):
    return [type(exc) for exc in info["exception"]]


class TestIssue777Simulate(unittest.TestCase):
    """`obs.simulate`: the injections are compensated relatively to the forecasts of the loads"""

    LOAD_ID = 1

    def setUp(self):
        self.env = _make_env("educ_case14_redisp", type(self).__name__)
        self.obs = self.env.reset(seed=0, options={"time serie id": 0})
        sim_obs, _, done, info = self.obs.simulate(self.env.action_space({}))
        assert not done, info["exception"]
        self.forecast = sim_obs.load_p[self.LOAD_ID]

    def tearDown(self):
        self.env.close()

    def _simulate(self, delta, obs=None):
        obs = self.obs if obs is None else obs
        return obs.simulate(_injection(self.env, self.LOAD_ID, self.forecast + delta))

    def test_no_change_needs_no_dispatch(self):
        sim_obs, _, done, info = self._simulate(0.0)
        assert not done, info["exception"]
        assert abs(sim_obs.actual_dispatch.sum()) <= 1e-3

    def test_increase_is_compensated(self):
        sim_obs, _, done, info = self._simulate(6.0)
        assert not done, info["exception"]
        assert abs(sim_obs.load_p[self.LOAD_ID] - self.forecast - 6.0) <= 1e-2
        assert abs(sim_obs.actual_dispatch.sum() - 6.0) <= 1e-2

    def test_decrease_is_compensated(self):
        sim_obs, _, done, info = self._simulate(-6.0)
        assert not done, info["exception"]
        assert abs(sim_obs.actual_dispatch.sum() + 6.0) <= 1e-2

    def test_too_large_increase_ends_the_simulation(self):
        delta = 2.0 * type(self.env).gen_max_ramp_up.sum()
        _, _, done, info = self._simulate(delta)
        assert done
        assert ImpossibleRedispatching in _exceptions(info)

    def test_simulate_does_not_change_the_environment(self):
        _ = self._simulate(6.0)
        assert self.env._dispatch_state.load_injection_excess.sum() == 0.0
        # the real step is the same as if there was no simulation before
        obs, _, done, info = self.env.step(self.env.action_space({}))
        assert not done, info["exception"]
        assert abs(obs.actual_dispatch.sum()) <= 1e-3

    def test_injection_of_the_real_environment_ends_in_the_simulation(self):
        # injection in the environment: the generators are 6 MW above the time series ...
        real_load = self.env.step(self.env.action_space({}))[0].load_p[self.LOAD_ID]
        obs = self.env.reset(seed=0, options={"time serie id": 0})
        obs, _, done, info = self.env.step(_injection(self.env, self.LOAD_ID, real_load + 6.0))
        assert not done, info["exception"]
        assert abs(obs.actual_dispatch.sum() - 6.0) <= 1e-2
        # ... and they come back when the next step is simulated (the loads come from the forecasts)
        sim_obs, _, done, info = obs.simulate(self.env.action_space({}))
        assert not done, info["exception"]
        assert abs(sim_obs.actual_dispatch.sum()) <= 1e-2


class TestIssue777Copy(unittest.TestCase):
    """`env.copy()`: the copy behaves like the original, and nothing is shared with it"""

    LOAD_ID = 1

    def setUp(self):
        self.env = _make_env("educ_case14_redisp", type(self).__name__)
        self.env.reset(seed=0, options={"time serie id": 0})
        self.load_ts = self.env.step(self.env.action_space({}))[0].load_p.copy()
        self.env.reset(seed=0, options={"time serie id": 0})
        # a load modification before the copy
        obs, _, done, info = self.env.step(_injection(self.env, self.LOAD_ID, self.load_ts[self.LOAD_ID] + 6.0))
        assert not done, info["exception"]
        # consumption given by the time series for the next step
        reference = self.env.copy()
        self.load_next = reference.step(reference.action_space({}))[0].load_p.copy()
        reference.close()
        self.copy = self.env.copy()

    def tearDown(self):
        self.copy.close()
        self.env.close()

    def test_copy_keeps_the_state_of_the_injections(self):
        assert np.array_equal(
            self.copy._dispatch_state.load_injection_excess, self.env._dispatch_state.load_injection_excess
        )
        assert abs(self.copy._dispatch_state.load_injection_excess.sum() - 6.0) <= 1e-2

    def test_copy_does_not_share_the_data_of_the_injections(self):
        assert self.copy._dispatch_state.load_injection_excess is not self.env._dispatch_state.load_injection_excess
        # a modification of the load in the copy is not seen by the original environment
        before = self.env._dispatch_state.load_injection_excess.copy()
        _, _, done, info = self.copy.step(_injection(self.copy, self.LOAD_ID, self.load_ts[self.LOAD_ID] + 3.0))
        assert not done, info["exception"]
        assert np.array_equal(self.env._dispatch_state.load_injection_excess, before)
        # and the other way around
        before_copy = self.copy._dispatch_state.load_injection_excess.copy()
        _, _, done, info = self.env.step(_injection(self.env, self.LOAD_ID, self.load_ts[self.LOAD_ID] - 3.0))
        assert not done, info["exception"]
        assert np.array_equal(self.copy._dispatch_state.load_injection_excess, before_copy)

    def _same_result(self, make_action):
        obs_env, _, done_env, info_env = self.env.step(make_action(self.env))
        obs_copy, _, done_copy, info_copy = self.copy.step(make_action(self.copy))
        assert done_env == done_copy
        assert _exceptions(info_env) == _exceptions(info_copy)
        assert np.allclose(obs_env.actual_dispatch, obs_copy.actual_dispatch)
        assert np.allclose(obs_env.gen_p, obs_copy.gen_p)
        assert np.allclose(obs_env.load_p, obs_copy.load_p)
        return obs_copy

    def test_normal_action_behaves_normally(self):
        obs = self._same_result(lambda env: env.action_space({}))
        # the injection of the step before the copy is over
        assert abs(obs.actual_dispatch.sum()) <= 1e-2

    def test_reduce_load_action_behaves_normally(self):
        obs = self._same_result(lambda env: _injection(env, self.LOAD_ID, self.load_next[self.LOAD_ID] - 4.0))
        assert abs(obs.actual_dispatch.sum() + 4.0) <= 1e-2

    def test_load_modification_after_the_copy_gives_the_same_results(self):
        obs = self._same_result(lambda env: _injection(env, self.LOAD_ID, self.load_next[self.LOAD_ID] + 5.0))
        assert abs(obs.actual_dispatch.sum() - 5.0) <= 1e-2

    def test_too_large_modification_after_the_copy_ends_the_episode(self):
        delta = 2.0 * type(self.copy).gen_max_ramp_up.sum()
        _, _, done, info = self.copy.step(_injection(self.copy, self.LOAD_ID, self.load_next[self.LOAD_ID] + delta))
        assert done
        assert ImpossibleRedispatching in _exceptions(info)


class _InjectionAgent(BaseAgent):
    """adds 2 MW to the load 1 at each step"""

    def act(self, observation, reward, done=False):
        load_p = np.full(type(self.action_space).n_load, fill_value=np.nan, dtype=dt_float)
        load_p[1] = observation.load_p[1] + 2.0
        return self.action_space({"injection": {"load_p": load_p}})


class TestIssue777Runner(unittest.TestCase):
    """The Runner gives the same results as a loop on the environment, and the dispatch follows the loads"""

    NB_STEP = 6

    def setUp(self):
        self.env = _make_env("educ_case14_redisp", type(self).__name__)

    def tearDown(self):
        self.env.close()

    def _run(self, agent_class, path_save):
        runner = Runner(**self.env.get_params_for_runner(), agentClass=agent_class)
        res = runner.run(nb_episode=1, max_iter=self.NB_STEP, env_seeds=[0], episode_id=[0], nb_process=1,
                         path_save=path_save)
        return res, EpisodeData.from_disk(path_save, res[0][1])

    def test_runner_dispatch_follows_the_changes_of_the_loads(self):
        with tempfile.TemporaryDirectory() as path_ref, tempfile.TemporaryDirectory() as path:
            _, episode_ref = self._run(DoNothingAgent, path_ref)
            res, episode = self._run(_InjectionAgent, path)
        assert res[0][3] == self.NB_STEP
        # the only difference with the time series is the injection: it is exactly what the generators compensate
        for obs_ref, obs in zip(episode_ref.observations, episode.observations):
            load_change = obs.load_p.sum() - obs_ref.load_p.sum()
            assert abs(obs.actual_dispatch.sum() - load_change) <= 1e-2
        assert abs(episode.observations[-1].actual_dispatch.sum()) > 1.0

    def test_runner_gives_the_same_results_as_a_loop(self):
        with tempfile.TemporaryDirectory() as path:
            res, episode = self._run(_InjectionAgent, path)
        # same agent, same env, same seed and chronics, without the Runner
        env = _make_env("educ_case14_redisp", type(self).__name__ + "_loop")
        try:
            env.seed(0)
            env.set_id(0)
            obs = env.reset()
            agent = _InjectionAgent(env.action_space)
            rewards = []
            dispatch = [obs.actual_dispatch.sum()]
            for _ in range(self.NB_STEP):
                obs, reward, done, info = env.step(agent.act(obs, 0.0))
                assert not done, info["exception"]
                rewards.append(reward)
                dispatch.append(obs.actual_dispatch.sum())
        finally:
            env.close()
        # the Runner stores 0. as the reward of the last step of the episode
        assert np.allclose(episode.rewards[:-1], rewards[:-1], atol=1e-2)
        assert abs(res[0][2] - sum(rewards[:-1])) <= 1e-2
        assert np.allclose(dispatch, [obs.actual_dispatch.sum() for obs in episode.observations], atol=1e-2)


class TestIssue777ForecastEnv(unittest.TestCase):
    """`obs.get_forecast_env()`: this environment behaves like the others"""

    LOAD_ID = 3

    def setUp(self):
        # the forecast environment of the small environments only has one step
        self.env = _make_env("l2rpn_idf_2023", type(self).__name__)
        self.obs = self.env.reset(seed=0, options={"time serie id": 0})
        forecast_env = self.obs.get_forecast_env()
        obs_ref, _, done, info = forecast_env.step(forecast_env.action_space({}))
        assert not done, info["exception"]
        self.forecast = obs_ref.load_p[self.LOAD_ID]
        forecast_env.close()

    def tearDown(self):
        self.env.close()

    def test_normal_action_behaves_normally(self):
        forecast_env = self.obs.get_forecast_env()
        obs, _, done, info = forecast_env.step(forecast_env.action_space({}))
        assert not done, info["exception"]
        assert abs(obs.actual_dispatch.sum()) <= 1e-3
        forecast_env.close()

    def test_increase_is_compensated(self):
        forecast_env = self.obs.get_forecast_env()
        obs, _, done, info = forecast_env.step(_injection(forecast_env, self.LOAD_ID, self.forecast + 3.0))
        assert not done, info["exception"]
        assert abs(obs.load_p[self.LOAD_ID] - self.forecast - 3.0) <= 1e-2
        assert abs(obs.actual_dispatch.sum() - 3.0) <= 1e-2
        forecast_env.close()

    def test_decrease_is_compensated(self):
        forecast_env = self.obs.get_forecast_env()
        obs, _, done, info = forecast_env.step(_injection(forecast_env, self.LOAD_ID, self.forecast - 3.0))
        assert not done, info["exception"]
        assert abs(obs.actual_dispatch.sum() + 3.0) <= 1e-2
        forecast_env.close()

    def test_too_large_increase_ends_the_episode(self):
        forecast_env = self.obs.get_forecast_env()
        delta = 2.0 * type(forecast_env).gen_max_ramp_up.sum()
        _, _, done, info = forecast_env.step(_injection(forecast_env, self.LOAD_ID, self.forecast + delta))
        assert done
        assert ImpossibleRedispatching in _exceptions(info)
        forecast_env.close()

    def test_dispatch_goes_back_to_zero_when_the_injection_stops(self):
        forecast_env = self.obs.get_forecast_env()
        _ = forecast_env.step(_injection(forecast_env, self.LOAD_ID, self.forecast + 3.0))
        obs, _, done, info = forecast_env.step(forecast_env.action_space({}))
        assert not done, info["exception"]
        assert abs(obs.actual_dispatch.sum()) <= 1e-2
        forecast_env.close()


if __name__ == "__main__":
    unittest.main()
