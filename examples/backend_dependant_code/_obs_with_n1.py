# Copyright (c) 2025, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import copy
from typing import Literal
import numpy as np
from grid2op.Environment.baseEnv import BaseEnv
from grid2op.Observation import CompleteObservation
from grid2op.Backend import Backend
from grid2op.dtypes import dt_float


class ObsWithN1(CompleteObservation):
    # attributes that will be saved when action is
    # serialized as a numpy vector
    attr_list_vect = copy.deepcopy(CompleteObservation.attr_list_vect)
    attr_list_vect.append("n1_vals")
    
    # attributes that will also be used when action is
    # serialized as json
    attr_list_json = copy.deepcopy(CompleteObservation.attr_list_json)
    
    # attributes that will be copied
    # when observation is copied
    attr_vect_cpy  = copy.deepcopy(CompleteObservation.attr_vect_cpy)
    attr_vect_cpy.append("n1_vals")
    
    def __init__(self,
                 obs_env=None,
                 action_helper=None,
                 random_prng=None,
                 kwargs_env=None,
                 n1_li=None,
                 reduce_n1: Literal["max", "count", "sum"]="max",
                 compute_algo: Literal["ac", "dc"]="ac"):
        super().__init__(obs_env,
                         action_helper,
                         random_prng,
                         kwargs_env,
                         n1_li=n1_li,
                         reduce_n1=reduce_n1,
                         compute_algo=compute_algo)
        
        # list of lines for which to compute the n-1
        if n1_li is None:
            # all n-1 will be used
            self._n1_li = np.arange(type(self).n_line, dtype=int)
        else:
            self._n1_li = []
            for el in n1_li:
                if isinstance(el, str):
                    # user provided a line name
                    el = type(self).get_line_info(line_name=el)[0]
                else:
                    # user provided a line id
                    el = int(el)
                self._n1_li.append(el)
            # convert it to np array
            self._n1_li = np.array(self._n1_li, dtype=int)
        
        # function to aggregate all information for one n-1 
        # into a single scalar
        self._fun_reduce_n1 = reduce_n1
        
        # computation method (AC or DC)
        self._compute_algo = compute_algo
        
        # added atributes
        self.n1_vals = np.empty(self._n1_li.shape, dtype=dt_float)
            
    def update(self, env: BaseEnv, with_forecast=True):
        # update standard attribute
        super().update(env, with_forecast=with_forecast)
        
        # update n1 attribute (specific to this usecase)
        for id_, line_id in enumerate(self._n1_li):
            this_backend : Backend = env.backend.copy_public()
            this_backend._disconnect_line(line_id)
            if self._compute_algo == "ac":
                conv, exc_ = this_backend.runpf(is_dc=False)
            elif self._compute_algo == "dc":
                conv, exc_ = this_backend.runpf(is_dc=True)
            else:
                raise RuntimeError(f"Unknown algorithm method '{self._compute_algo}', "
                                   "use one of 'ac' or 'dc'")
            if not conv:
                # powerflow has diverged
                self.n1_vals[id_] = np.nan
                continue
            
            rel_flow = this_backend.get_relative_flow()[self._n1_li]
            is_finite = np.isfinite(rel_flow)
            is_infinite = ~is_finite
            
            if self._fun_reduce_n1 == "max":
                if is_infinite.any():
                    self.n1_vals[id_] = 5. # some kind of "gentle" max
                else:
                    self.n1_vals[id_] = rel_flow[is_finite].max()
            elif self._fun_reduce_n1 == "count":
                self.n1_vals[id_] = (rel_flow[is_finite] >= 1.).sum()
                self.n1_vals[id_] += is_infinite.sum() 
            elif self._fun_reduce_n1 == "sum":
                self.n1_vals[id_] = rel_flow[is_finite].sum()
                self.n1_vals[id_] += is_infinite.sum() * 5.
            else:
                raise RuntimeError("Unknown way to summarize information for each n1, found "
                                   f"'{self._fun_reduce_n1}', please use one of "
                                   "'max', 'count' or 'sum'")
                