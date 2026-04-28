# Copyright (c) 2026, RTE (https://www.rte-france.com)
# See AUTHORS.txt and https://github.com/Grid2Op/grid2op/pull/319
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import warnings
import grid2op
import numpy as np
import unittest

from lightsim2grid import LightSimBackend

import grid2op
import numpy as np
from datetime import timedelta
from lightsim2grid import LightSimBackend
from grid2op.Chronics import FromNPY
from grid2op.Exceptions import ChronicsError
from typing import Optional


class FromNPYMultiHorizon(FromNPY):
    """
    Extension of FromNPY supporting multi-horizon forecasts.

    Forecast arrays can be:
        (T, N)         -> automatically converted to (T, 1, N)
        (T, H, N)      -> used directly

    Where:
        T = number of timesteps
        H = number of forecast horizons
        N = number of loads / generators
    """

    MULTI_CHRONICS = False

    def __init__(
        self,
        *args,
        load_p_forecast: Optional[np.ndarray] = None,
        load_q_forecast: Optional[np.ndarray] = None,
        prod_p_forecast: Optional[np.ndarray] = None,
        prod_v_forecast: Optional[np.ndarray] = None,
        **kwargs,
    ):
        # Prevent parent from building recursive forecast object
        super().__init__(
            *args,
            load_p_forecast=None,
            load_q_forecast=None,
            prod_p_forecast=None,
            prod_v_forecast=None,
            **kwargs,
        )

        # Format forecast arrays
        self._load_p_forecast = self._format_forecast_array(load_p_forecast)
        self._load_q_forecast = self._format_forecast_array(load_q_forecast)
        self._prod_p_forecast = self._format_forecast_array(prod_p_forecast)
        self._prod_v_forecast = self._format_forecast_array(prod_v_forecast)

        # Determine number of horizons
        if self._load_p_forecast is not None:
            self.n_forecast_horizons = self._load_p_forecast.shape[1]
        else:
            self.n_forecast_horizons = 0

    @staticmethod
    def _format_forecast_array(arr: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """
        Convert forecast array to shape (T, H, N)

        Accepts:
            (T, N)
            (T, H, N)
        """
        if arr is None:
            return None

        if arr.ndim == 2:
            # (T, N) → (T, 1, N)
            return arr[:, np.newaxis, :]

        if arr.ndim == 3:
            return arr

        raise ChronicsError("Forecast arrays must have shape (T, N) or (T, H, N)")

    def forecasts(self):
        """
        Return list of (datetime, dict) for each horizon.
        """
        if self._load_p_forecast is None:
            return []

        t = self.current_index

        if t >= self._load_p_forecast.shape[0]:
            return []

        results = []

        for h in range(self.n_forecast_horizons):
            dict_ = {}

            if self._load_p_forecast is not None:
                dict_["load_p"] = 1.0 * self._load_p_forecast[t, h, :]

            if self._load_q_forecast is not None:
                dict_["load_q"] = 1.0 * self._load_q_forecast[t, h, :]

            if self._prod_p_forecast is not None:
                dict_["prod_p"] = 1.0 * self._prod_p_forecast[t, h, :]

            if self._prod_v_forecast is not None:
                dict_["prod_v"] = 1.0 * self._prod_v_forecast[t, h, :]

            forecast_datetime = self.current_datetime + (h + 1) * self.time_interval

            results.append((forecast_datetime, {"injection": dict_}))

        return results

    def change_forecasts(
        self,
        new_load_p: Optional[np.ndarray] = None,
        new_load_q: Optional[np.ndarray] = None,
        new_prod_p: Optional[np.ndarray] = None,
        new_prod_v: Optional[np.ndarray] = None,
    ):
        """
        Update forecast arrays (effective after env.reset()).
        """
        if new_load_p is not None:
            self._load_p_forecast = self._format_forecast_array(new_load_p)

        if new_load_q is not None:
            self._load_q_forecast = self._format_forecast_array(new_load_q)

        if new_prod_p is not None:
            self._prod_p_forecast = self._format_forecast_array(new_prod_p)

        if new_prod_v is not None:
            self._prod_v_forecast = self._format_forecast_array(new_prod_v)

        if self._load_p_forecast is not None:
            self.n_forecast_horizons = self._load_p_forecast.shape[1]
        else:
            self.n_forecast_horizons = 0

    def check_validity(self, backend=None):
        super().check_validity(backend)

        if self._load_p_forecast is not None:
            assert self._load_p_forecast.shape[0] == self._load_p.shape[0]
            assert self._load_p_forecast.shape[2] == self.n_load

        if self._prod_p_forecast is not None:
            assert self._prod_p_forecast.shape[0] == self._prod_p.shape[0]
            assert self._prod_p_forecast.shape[2] == self.n_gen
            
            
class TestIssueL2G128(unittest.TestCase):
    # test the https://github.com/Grid2op/lightsim2grid/issues/128 issue
    # which is actually a grid2Op issue
    def setUp(self):
        # Artificially create a timeseries :
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = grid2op.make("l2rpn_case14_sandbox", test=True, backend=LightSimBackend())
        init_obs = env.reset(seed=0, options={"time serie id": 0})
        
        # artificially generate time series
        load_p, load_q, prod_p, prod_v = (
            init_obs.load_p,
            init_obs.load_q,
            init_obs.prod_p,
            init_obs.prod_v,
        )

        load_p = np.concatenate([[load_p] * 48], axis=0)
        load_q = np.concatenate([[load_q] * 48], axis=0)
        prod_p = np.concatenate([[prod_p] * 48], axis=0)
        prod_v = np.concatenate([[prod_v] * 48], axis=0)

        load_p_fc = load_p
        load_q_fc = load_q
        prod_p_fc = prod_p
        prod_v_fc = prod_v

        # create a fake moving timeserie that will create a DC divergence (too high load)

        load_p_fake = (load_p + (np.linspace(0, 1, len(load_p)) * 100).reshape(-1, 1)).round(1)

        T, N = load_p_fake.shape
        H = 48

        indices = np.arange(T)[:, None] + np.arange(1, H + 1)[None, :]
        indices = np.clip(indices, 0, T - 1)

        load_p_fc = load_p_fake[indices]
        load_q_fc = load_q[indices]
        prod_p_fc = prod_p[indices]
        prod_v_fc = prod_v[indices]

        load_p_fc[1:, :, :] = (
            0  # Forecast available at t =0. We only need the first row for each forecast
        )
        
        self.env = grid2op.make(
            "l2rpn_case14_sandbox",
            backend=LightSimBackend(),
            allow_detachment=True,
            n_busbar=6,
            chronics_class=FromNPYMultiHorizon,
            data_feeding_kwargs={
                "i_start": 0,
                "i_end": 48,
                "load_p": load_p_fake,
                "load_q": load_q,
                "prod_p": prod_p,
                "prod_v": prod_v,
                "load_p_forecast": load_p_fc,
                "load_q_forecast": load_q_fc,
                "prod_p_forecast": prod_p_fc,
                "prod_v_forecast": prod_v_fc,
                "h_forecast": [i * 30 for i in range(1, 48)],
                "time_interval": timedelta(minutes=30),
            },
        )


        self.env.chronics_handler.change_chronics(
            new_load_p=load_p_fake,
            new_load_q=load_q,
            new_prod_p=prod_p,
            new_prod_v=prod_v,
        )
        self.env.chronics_handler.change_forecasts(
            new_load_p=load_p_fc,
            new_load_q=load_q_fc,
            new_prod_p=prod_p_fc,
            new_prod_v=prod_v_fc,
        )
        return super().setUp()

    def test_simulate_works(self):
        obs_init = self.env.reset()
        # obs ok
        obs_simulate, _, done, info = obs_init.simulate(self.env.action_space(), time_step=5)
        assert not done
        load_p_real = obs_simulate.load_p.copy()
        rho_real = obs_simulate.rho.copy()
        # lead to a game over
        obs_simulate, _, done, info = obs_init.simulate(self.env.action_space(), time_step=9)
        assert done
        # resimulate the time_step = 5, issue #128 => lead to a game over
        obs_simulate, _, done, info = obs_init.simulate(self.env.action_space(), time_step=5)
        assert not done
        assert np.allclose(load_p_real, obs_simulate.load_p)
        assert np.allclose(rho_real, obs_simulate.rho)
