# Copyright (c) 2019-2026, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import unittest
import warnings
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import grid2op
from grid2op.Action import PowerlineSetAction
from grid2op.Exceptions import OpponentError
from grid2op.Opponent import BaseActionBudget, LookaheadLineOpponent, WeightedRandomOpponent

# lines of l2rpn_case14_sandbox whose disconnection does not split the grid
LINES_ATTACKED = ["0_1_0", "0_4_1", "1_2_2", "1_3_3", "1_4_4", "3_4_6", "5_10_7", "8_9_10"]
ATTACK_PERIOD = 12
ATTACK_DURATION = 6


def _make_env(name, opponent_class, kwargs_opponent):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        return grid2op.make("l2rpn_case14_sandbox",
                            test=True,
                            _add_to_name=name,
                            opponent_attack_cooldown=ATTACK_PERIOD,
                            opponent_attack_duration=ATTACK_DURATION,
                            opponent_budget_per_ts=1.0,
                            opponent_init_budget=100.,
                            opponent_action_class=PowerlineSetAction,
                            opponent_class=opponent_class,
                            opponent_budget_class=BaseActionBudget,
                            kwargs_opponent=kwargs_opponent)


def _new_attacks(env, seed, nb_step=120):
    """Do nothing for at most `nb_step` steps. Return the (step, line id) of each new attack and the
    number of steps played."""
    env.seed(seed)
    env.reset()
    res = []
    step = 0
    for step in range(nb_step):
        _, _, done, info = env.step(env.action_space())
        line = info["opponent_attack_line"]
        if line is not None and info["opponent_attack_duration"] == ATTACK_DURATION:
            res.append((step, int(np.flatnonzero(line)[0])))
        if done:
            break
    return res, step


class TestIndependentTiming(unittest.TestCase):
    def _run(self, opponent_class, extra, seed=1):
        kwargs = {"lines_attacked": LINES_ATTACKED, "attack_period": ATTACK_PERIOD, **extra}
        env = _make_env(f"{type(self).__name__}_{opponent_class.__name__}", opponent_class, kwargs)
        try:
            return _new_attacks(env, seed)
        finally:
            env.close()

    def test_same_attack_steps_for_different_line_choices(self):
        weighted, weighted_steps = self._run(WeightedRandomOpponent,
                                             {"independent_timing": True,
                                              "rho_normalization": [1.0] * len(LINES_ATTACKED)})
        lookahead, lookahead_steps = self._run(LookaheadLineOpponent, {"independent_timing": True})
        horizon = min(weighted_steps, lookahead_steps)
        weighted_steps_list = [step for step, _ in weighted if step <= horizon]
        # a shared random generator would only show up from the second attack on
        self.assertGreaterEqual(len(weighted_steps_list), 2)
        self.assertEqual(weighted_steps_list, [step for step, _ in lookahead if step <= horizon])

    def test_opponent_simulations_are_not_counted_as_the_agent_ones(self):
        env = _make_env(f"{type(self).__name__}_counter", LookaheadLineOpponent,
                        {"lines_attacked": LINES_ATTACKED, "attack_period": ATTACK_PERIOD})
        try:
            attacks, _ = _new_attacks(env, seed=1)
            self.assertGreater(len(attacks), 0)
            self.assertEqual(env.nb_highres_called, 0)
        finally:
            env.close()

    def test_copy_keeps_the_timing_state(self):
        kwargs = {"lines_attacked": LINES_ATTACKED, "attack_period": ATTACK_PERIOD, "independent_timing": True,
                  "rho_normalization": [1.0] * len(LINES_ATTACKED)}
        env = _make_env(f"{type(self).__name__}_copy", WeightedRandomOpponent, kwargs)
        try:
            env.seed(3)
            env.reset()
            for _ in range(5):
                env.step(env.action_space())
            env_cpy = env.copy()
            res, res_cpy = [], []
            for _ in range(60):
                _, _, done, info = env.step(env.action_space())
                _, _, done_cpy, info_cpy = env_cpy.step(env_cpy.action_space())
                res.append(info["opponent_attack_line"] is not None)
                res_cpy.append(info_cpy["opponent_attack_line"] is not None)
                self.assertEqual(done, done_cpy)
                if done:
                    break
            self.assertEqual(res, res_cpy)
            env_cpy.close()
        finally:
            env.close()


class TestLookaheadChoice(unittest.TestCase):
    def setUp(self) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make("l2rpn_case14_sandbox", test=True, _add_to_name=type(self).__name__)
        self.obs = self.env.reset(seed=0, options={"time serie id": 0})
        self.line_ids = [int(np.flatnonzero(self.env.name_line == name)[0]) for name in LINES_ATTACKED]

    def tearDown(self) -> None:
        self.env.close()

    def _opponent(self):
        opp = LookaheadLineOpponent(self.env.action_space)
        opp.init(self.env, lines_attacked=LINES_ATTACKED)
        opp.seed(0)
        opp.reset(0.)
        return opp

    @staticmethod
    def _attacked_line(attack):
        return int(np.flatnonzero(attack._set_line_status == -1)[0])

    def _expected_line(self, obs):
        simulator = obs.get_simulator()
        best, best_score = None, -np.inf
        for l_id in self.line_ids:
            if not obs.line_status[l_id]:
                continue
            res = simulator.predict(act=self.env.action_space({"set_line_status": [(l_id, -1)]}), do_copy=True)
            score = np.inf if not res.converged else float(np.max(res.current_obs.rho))
            if best is None or score > best_score + 1e-9:
                best, best_score = l_id, score
        return best

    def test_picks_the_line_with_the_highest_simulated_rho(self):
        opp = self._opponent()
        status = self.obs.line_status[opp._lines_ids]
        self.assertEqual(self._attacked_line(opp._choose_attack(self.obs, status)), self._expected_line(self.obs))

    def test_skips_disconnected_lines(self):
        opp = self._opponent()
        best = self._expected_line(self.obs)
        obs, *_ = self.env.step(self.env.action_space({"set_line_status": [(best, -1)]}))
        self.assertFalse(obs.line_status[best])
        chosen = self._attacked_line(opp._choose_attack(obs, obs.line_status[opp._lines_ids]))
        self.assertNotEqual(chosen, best)
        self.assertEqual(chosen, self._expected_line(obs))

    def _stub_simulator(self, scores):
        """A simulator whose prediction for the disconnection of line `l` has max rho `scores[l]`,
        or does not converge when `scores[l]` is None."""
        n_line = self.env.n_line

        def predict(act, do_copy=True):
            l_id = self._attacked_line(act)
            if scores[l_id] is None:
                return SimpleNamespace(converged=False, current_obs=None)
            return SimpleNamespace(converged=True, current_obs=SimpleNamespace(rho=np.full(n_line, scores[l_id])))

        return SimpleNamespace(predict=predict)

    def test_a_diverging_power_flow_is_the_strongest_attack(self):
        opp = self._opponent()
        scores = {l_id: 0.5 for l_id in self.line_ids}
        scores[self.line_ids[3]] = None
        status = self.obs.line_status[opp._lines_ids]
        with patch.object(self.obs, "get_simulator", return_value=self._stub_simulator(scores)):
            self.assertEqual(self._attacked_line(opp._choose_attack(self.obs, status)), self.line_ids[3])

    def _choose_with_scores(self, opp, scores):
        status = self.obs.line_status[opp._lines_ids]
        with patch.object(self.obs, "get_simulator", return_value=self._stub_simulator(scores)):
            return self._attacked_line(opp._choose_attack(self.obs, status))

    def test_ties_go_to_the_first_listed_line(self):
        opp = self._opponent()
        scores = {l_id: 0.5 for l_id in self.line_ids}
        self.assertEqual(self._choose_with_scores(opp, scores), self.line_ids[0])
        # a later line within the tolerance does not replace the first one
        scores[self.line_ids[2]] = 0.5 + 5e-10
        self.assertEqual(self._choose_with_scores(opp, scores), self.line_ids[0])
        # a later line above the tolerance does
        scores[self.line_ids[2]] = 0.5 + 2e-9
        self.assertEqual(self._choose_with_scores(opp, scores), self.line_ids[2])

    def test_choice_uses_no_random_number(self):
        opp = self._opponent()
        before = opp.space_prng.get_state()
        opp._choose_attack(self.obs, self.obs.line_status[opp._lines_ids])
        after = opp.space_prng.get_state()
        self.assertTrue(np.array_equal(before[1], after[1]))
        self.assertEqual(before[2], after[2])

    def test_clear_error_without_simulator(self):
        opp = self._opponent()
        self.env.deactivate_forecast()
        obs = self.env.reset(seed=0, options={"time serie id": 0})
        with self.assertRaises(OpponentError):
            opp._choose_attack(obs, obs.line_status[opp._lines_ids])


if __name__ == "__main__":
    unittest.main()
