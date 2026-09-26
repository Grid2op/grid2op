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
from grid2op.Agent import DoNothingAgent
from grid2op.Environment.dispatch import BaseRedispatchSolver, DefaultRedispatchSolver
from grid2op.Exceptions import EnvError, ImpossibleRedispatching
from grid2op.Runner import Runner


class CountingSolver(BaseRedispatchSolver):
    """A solver that only implements the abstract API: it delegates to the default one
    and counts how many times it has been called."""
    def __init__(self):
        super().__init__()
        self.nb_call = 0
        self.nb_reset = 0
        self._default = DefaultRedispatchSolver()

    def solve(self, constraints, state):
        self.nb_call += 1
        return self._default.solve(constraints, state)

    def reset(self):
        self.nb_reset += 1


class FailingSolver(BaseRedispatchSolver):
    """A solver that never finds a dispatch."""
    def solve(self, constraints, state):
        return ImpossibleRedispatching("no dispatch from FailingSolver")


class TestCustomRedispatchSolver(unittest.TestCase):
    def _make(self, redispatch_solver):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            return grid2op.make("l2rpn_case14_sandbox",
                                test=True,
                                redispatch_solver=redispatch_solver,
                                _add_to_name=type(self).__name__)

    def setUp(self) -> None:
        self.solver = CountingSolver()
        self.env = self._make(self.solver)
        self.obs = self.env.reset(seed=0, options={"time serie id": 0})
        self.gen_id = int(np.nonzero(type(self.env).gen_redispatchable)[0][0])
        return super().setUp()

    def tearDown(self) -> None:
        self.env.close()
        return super().tearDown()

    def _redisp_act(self, env):
        return env.action_space({"redispatch": [(self.gen_id, 1.)]})

    def test_env_uses_a_copy_of_the_solver(self):
        solver = self.env._redispatch_solver
        assert isinstance(solver, CountingSolver)
        # the instance given to make is not modified, each env has its own copy
        assert solver is not self.solver
        assert self.solver.env is None
        assert solver.env is self.env

    def test_step(self):
        solver = self.env._redispatch_solver
        nb_reset = solver.nb_reset
        obs, reward, done, info = self.env.step(self._redisp_act(self.env))
        assert not done, info["exception"]
        assert not info["exception"], info["exception"]
        assert solver.nb_call >= 1
        assert abs(obs.target_dispatch[self.gen_id] - 1.) <= 1e-5
        assert abs(obs.actual_dispatch.sum()) <= 1e-3
        self.env.reset(seed=0, options={"time serie id": 0})
        assert solver.nb_reset == nb_reset + 1

    def test_class_is_accepted(self):
        env = self._make(CountingSolver)
        try:
            assert isinstance(env._redispatch_solver, CountingSolver)
            env.reset(seed=0, options={"time serie id": 0})
            obs, reward, done, info = env.step(self._redisp_act(env))
            assert not done, info["exception"]
        finally:
            env.close()

    def test_invalid_solver(self):
        with self.assertRaises(EnvError):
            self._make(DefaultRedispatchSolver.solve)
        with self.assertRaises(EnvError):
            self._make(int)

    def test_failing_solver_is_game_over(self):
        env = self._make(FailingSolver())
        try:
            env.reset(seed=0, options={"time serie id": 0})
            obs, reward, done, info = env.step(self._redisp_act(env))
            assert done
            assert any("no dispatch from FailingSolver" in str(exc) for exc in info["exception"])
        finally:
            env.close()

    def test_simulate(self):
        obs_env = self.env.observation_space.obs_env
        assert isinstance(obs_env._redispatch_solver, CountingSolver)
        assert obs_env._redispatch_solver is not self.env._redispatch_solver
        sim_obs, sim_r, sim_d, sim_i = self.obs.simulate(self._redisp_act(self.env))
        assert not sim_d, sim_i["exception"]
        assert obs_env._redispatch_solver.nb_call >= 1

    def test_forecast_env(self):
        for_env = self.obs.get_forecast_env()
        try:
            assert isinstance(for_env._redispatch_solver, CountingSolver)
            for_env.reset()
            for_obs, *_ = for_env.step(self._redisp_act(for_env))
            assert for_env._redispatch_solver.nb_call >= 1
        finally:
            for_env.close()

    def test_copy(self):
        env_cpy = self.env.copy()
        try:
            assert isinstance(env_cpy._redispatch_solver, CountingSolver)
            assert env_cpy._redispatch_solver is not self.env._redispatch_solver
            assert env_cpy._redispatch_solver.env is env_cpy
            assert isinstance(env_cpy.observation_space.obs_env._redispatch_solver, CountingSolver)
        finally:
            env_cpy.close()

    def test_runner(self):
        params = self.env.get_params_for_runner()
        assert isinstance(params["redispatch_solver"], CountingSolver)
        assert params["redispatch_solver"].env is None
        runner = Runner(**params, agentClass=DoNothingAgent)
        env_runner, _ = runner._new_env(self.env.parameters)
        try:
            assert isinstance(env_runner._redispatch_solver, CountingSolver)
        finally:
            env_runner.close()
        res = runner.run(nb_episode=1, max_iter=3)
        assert len(res) == 1


class TestGenRedispatchableNotModified(unittest.TestCase):
    """When a detached generator makes the dispatch infeasible, the solver tries to use all
    the redispatchable generators. It used to remove the detached generators from the class
    attribute `gen_redispatchable` (shared by all the envs) while doing so."""
    def setUp(self) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make("educ_case14_storage",
                                    test=True,
                                    allow_detachment=True,
                                    action_class=CompleteAction,
                                    _add_to_name=type(self).__name__)
        params = self.env.parameters
        params.IGNORE_MIN_UP_DOWN_TIME = True
        params.ALLOW_DISPATCH_GEN_SWITCH_OFF = True
        self.env.change_parameters(params)
        self.env.reset(seed=0, options={"time serie id": 1})
        return super().setUp()

    def tearDown(self) -> None:
        self.env.close()
        return super().tearDown()

    def test_gen_redispatchable_unchanged(self):
        cls = type(self.env)
        gen_redisp_before = cls.gen_redispatchable.copy()
        gen_id = int(np.nonzero(gen_redisp_before)[0][0])
        obs, reward, done, info = self.env.step(self.env.action_space(
            {"set_bus": {"generators_id": [(gen_id, -1)]}}
        ))
        # detaching this generator is too much for the ramps of the others
        assert done
        assert any(isinstance(exc, ImpossibleRedispatching) for exc in info["exception"])
        assert (cls.gen_redispatchable == gen_redisp_before).all()
        assert (type(self.env.action_space).gen_redispatchable == gen_redisp_before).all()


if __name__ == "__main__":
    unittest.main()
