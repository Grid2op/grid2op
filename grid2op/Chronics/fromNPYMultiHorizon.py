# Copyright (c) 2026, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

from typing import Optional
import hashlib

import numpy as np

from grid2op.Chronics.fromNPY import FromNPY
from grid2op.Exceptions import ChronicsError


class FromNPYMultiHorizon(FromNPY):
    """FromNPY variant supporting one or multiple forecast horizons.

    Forecast arrays can use either the existing two-dimensional shape ``(T, N)``
    or a three-dimensional shape ``(T, H, N)``, where ``T`` is the number of
    timesteps, ``H`` the number of forecast horizons, and ``N`` the number of
    loads or generators. Two-dimensional forecasts are treated as one-horizon
    forecasts, which preserves the existing behaviour.
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
        # Let FromNPY manage the actual chronics while keeping forecast handling
        # in this class. This avoids the recursive one-step forecast object used
        # by the base implementation.
        super().__init__(
            *args,
            load_p_forecast=None,
            load_q_forecast=None,
            prod_p_forecast=None,
            prod_v_forecast=None,
            **kwargs,
        )

        self._load_p_forecast = self._format_forecast_array(load_p_forecast)
        self._load_q_forecast = self._format_forecast_array(load_q_forecast)
        self._prod_p_forecast = self._format_forecast_array(prod_p_forecast)
        self._prod_v_forecast = self._format_forecast_array(prod_v_forecast)

        self.__new_load_p_forecast = None
        self.__new_load_q_forecast = None
        self.__new_prod_p_forecast = None
        self.__new_prod_v_forecast = None

        self._update_n_forecast_horizons()
        self._check_forecast_arrays()

    @staticmethod
    def _format_forecast_array(
        arr: Optional[np.ndarray],
    ) -> Optional[np.ndarray]:
        if arr is None:
            return None
        if arr.ndim == 2:
            return 1.0 * arr[:, np.newaxis, :]
        if arr.ndim == 3:
            return 1.0 * arr
        raise ChronicsError(
            "Forecast arrays must have shape (T, N) or (T, H, N)."
        )

    def _update_n_forecast_horizons(self) -> None:
        if self._load_p_forecast is None:
            self.n_forecast_horizons = 0
        else:
            self.n_forecast_horizons = self._load_p_forecast.shape[1]

    def _check_forecast_arrays(self) -> None:
        if self._load_p_forecast is None:
            if self._load_q_forecast is not None or self._prod_p_forecast is not None:
                raise ChronicsError(
                    "load_p_forecast must be provided when load_q_forecast or "
                    "prod_p_forecast is provided."
                )
            if self._prod_v_forecast is not None:
                raise ChronicsError(
                    "load_p_forecast must be provided when prod_v_forecast is provided."
                )
            return

        if self._load_q_forecast is None:
            raise ChronicsError(
                "If you provide load_p_forecast you should provide load_q_forecast."
            )
        if self._prod_p_forecast is None:
            raise ChronicsError(
                "If you provide load_p_forecast you should provide prod_p_forecast."
            )

        expected_t = self._load_p.shape[0]
        expected_h = self._load_p_forecast.shape[1]

        arrays = (
            ("load_p_forecast", self._load_p_forecast, self.n_load),
            ("load_q_forecast", self._load_q_forecast, self.n_load),
            ("prod_p_forecast", self._prod_p_forecast, self.n_gen),
        )
        for name, arr, expected_n in arrays:
            if arr.shape[0] != expected_t:
                raise ChronicsError(
                    f"{name} must have the same number of timesteps as the chronics."
                )
            if arr.shape[1] != expected_h:
                raise ChronicsError(
                    "All forecast arrays must have the same number of horizons."
                )
            if arr.shape[2] != expected_n:
                raise ChronicsError(
                    f"{name} has an invalid number of columns for this environment."
                )

        if self._prod_v_forecast is not None:
            if self._prod_v_forecast.shape[0] != expected_t:
                raise ChronicsError(
                    "prod_v_forecast must have the same number of timesteps as the chronics."
                )
            if self._prod_v_forecast.shape[1] != expected_h:
                raise ChronicsError(
                    "All forecast arrays must have the same number of horizons."
                )
            if self._prod_v_forecast.shape[2] != self.n_gen:
                raise ChronicsError(
                    "prod_v_forecast has an invalid number of columns for this environment."
                )

    def _get_long_hash(self, hash_: Optional[hashlib.blake2b] = None):
        if hash_ is None:
            hash_ = hashlib.blake2b()
        super()._get_long_hash(hash_)
        for arr in (
            self._load_p_forecast,
            self._load_q_forecast,
            self._prod_p_forecast,
            self._prod_v_forecast,
        ):
            if arr is not None:
                hash_.update(arr.tobytes())
        return hash_.digest()

    def check_validity(self, backend=None) -> None:
        super().check_validity(backend)
        self._check_forecast_arrays()

    def forecasts(self):
        """Return one forecast entry per available horizon."""
        if self._load_p_forecast is None:
            return []

        t = self.current_index
        if t < 0 or t >= self._load_p_forecast.shape[0]:
            return []

        results = []
        for h in range(self.n_forecast_horizons):
            injection = {
                "load_p": 1.0 * self._load_p_forecast[t, h, :],
                "load_q": 1.0 * self._load_q_forecast[t, h, :],
                "prod_p": 1.0 * self._prod_p_forecast[t, h, :],
            }
            if self._prod_v_forecast is not None:
                injection["prod_v"] = 1.0 * self._prod_v_forecast[t, h, :]

            forecast_datetime = self.current_datetime + (h + 1) * self.time_interval
            results.append((forecast_datetime, {"injection": injection}))

        return results

    def change_forecasts(
        self,
        new_load_p: Optional[np.ndarray] = None,
        new_load_q: Optional[np.ndarray] = None,
        new_prod_p: Optional[np.ndarray] = None,
        new_prod_v: Optional[np.ndarray] = None,
    ) -> None:
        """Queue forecast changes to take effect after the next reset."""
        if new_load_p is not None:
            self.__new_load_p_forecast = self._format_forecast_array(new_load_p)
        if new_load_q is not None:
            self.__new_load_q_forecast = self._format_forecast_array(new_load_q)
        if new_prod_p is not None:
            self.__new_prod_p_forecast = self._format_forecast_array(new_prod_p)
        if new_prod_v is not None:
            self.__new_prod_v_forecast = self._format_forecast_array(new_prod_v)

    def next_chronics(self):
        super().next_chronics()

        if self.__new_load_p_forecast is not None:
            self._load_p_forecast = self.__new_load_p_forecast
            self.__new_load_p_forecast = None
        if self.__new_load_q_forecast is not None:
            self._load_q_forecast = self.__new_load_q_forecast
            self.__new_load_q_forecast = None
        if self.__new_prod_p_forecast is not None:
            self._prod_p_forecast = self.__new_prod_p_forecast
            self.__new_prod_p_forecast = None
        if self.__new_prod_v_forecast is not None:
            self._prod_v_forecast = self.__new_prod_v_forecast
            self.__new_prod_v_forecast = None

        self._update_n_forecast_horizons()
        self._check_forecast_arrays()
