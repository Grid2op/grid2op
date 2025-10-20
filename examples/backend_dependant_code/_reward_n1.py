# Copyright (c) 2025, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

from typing import Literal
import numpy as np
from grid2op.Reward import BaseReward
from grid2op.Environment import BaseEnv
from grid2op.Backend import Backend


class N1Reward(BaseReward):
    """This implements a "new" reward in grid2op.
    
    This reward will be based on the agregation of some data resulting of 
    the simulation of some contingencies.
    
    The kwargs `n1_li` is a list of contingencies to simulate. 
    If None (default), it will consist of all the lines of the grid.
    
    The kwargs `compute_algo` allows to specify if the way
    the contingencies are computed. It can be either "ac" or "dc".
    
    The kwargs `reduce_n1` and `reduce_reward` allows to specify how the information
    about the contingencies are agregated into a single number (reward).
    
    More specifically, say you are simulating contingency on line `i`, 
    then `reduce_n1` will be used to aggregate the flows on all the lines after
    this contingencies. It can be:
    - "max" (default) : the maximum flow on all the lines (after contingency `i`)
    - "count" : the number of lines that are overloaded (after contingency `i`)
    - "sum" : the sum of the flows on all the lines (after contingency `i`)
    
    Then `reduce_reward` will be used to aggregate to agregate information
    about all the contingencies. It can be:
    - "max" (default) : the maximum value of all the result of the previous
      computation for each contingency
    - "count" : the number of contingencies that have a result >= 1
    - "sum" : the sum of the result of the previous computation
    
    Examples:
    
    If `reduce_n1="max"` and `reduce_reward="max"` then the reward will be the maximum
    flow obtained on any line after a contingency. There are len(n1_li) such
    results. Then, these len(n1_li) results are aggregated and only the maximum is kept
    (`reduce_reward="max"`). => gives the maximum flows on any line after any contingency
    
    If `reduce_n1="count"` and `reduce_reward="sum"` then the reward will be the number
    of line on overflow for any contingency (`reduce_n1="count"`). These len(n1_li) 
    numbers will then be summed up to give the final reward (`reduce_reward="sum"`). => gives
    the total number of lines on overflow after any contingency
    
    Extra information: 
    
    **NB** This could be customized by counting only the overflows on a given set of lines (similar
    to `n1_li` but for "monitored lines" and not "disconnected lines").
    
    **NB** We did not assume anything on any previous comptuation here. But on some settings,
    the results of each contingency might be available in the observation (for example if using
    `ObsWithN1` observation class, with the same `n1_li` and the same `reduce_n1` and the 
    same `compute_algo`). In that
    case, instead of computing again the same contingencies, the reward can be computed by looking
    at the `obs.n1_vals` attribute. This optimization is not implemented here.
    
    """
    def __init__(self,
                 logger=None,
                 n1_li=None,
                 reduce_n1: Literal["max", "count", "sum"]="max",
                 reduce_reward: Literal["max", "count", "sum"]="max",
                 compute_algo: Literal["ac", "dc"]="ac"):
        super().__init__(logger)
        self.n1_li_init = n1_li
        self._n1_li = None
        
        # function to aggregate all information for one n-1 
        # into a single scalar
        self._fun_reduce_n1 = reduce_n1
        
        # function to aggregate all reward (for all n-1)
        self._fun_reduce_reward = reduce_reward
        
        # computation method (AC or DC)
        self._compute_algo = compute_algo
    
    def initialize(self, env: BaseEnv):
        super().initialize(env)
        if self.n1_li_init is None:
            self._n1_li = np.arange(type(env).n_line)
        else:
            self._n1_li = []
            for el in self.n1_li_init:
                if isinstance(el, str):
                    # user provided a line name
                    el = type(self).get_line_info(line_name=el)[0]
                else:
                    # user provided a line id
                    el = int(el)
                self._n1_li.append(el)
            # convert it to np array
            self._n1_li = np.array(self._n1_li, dtype=int)
            
    def __call__(self, action, env, has_error, is_done, is_illegal, is_ambiguous):
        if is_done and (not has_error and not is_illegal and not is_ambiguous):
            # episode terminated without error
            return 1. # max reward 
        if has_error:
            # episode truncated
            return -10. # min reward
        
        # TODO optimization here
        # if everything is the same in obs (provided that obs is ObsWithN1) and
        # in self (same lines simulated, same function to get the max)
        # then use:
        # obs = env.get_obs(_do_copy=False)
        # and obs.n1_vals instead of recomputing powerflows
        
        # perform the n1 computation
        n1_vals = np.zeros(self._n1_li.shape, dtype=float)
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
                n1_vals[id_] = 5.
                continue
            
            rel_flow = this_backend.get_relative_flow()[self._n1_li]
            is_finite = np.isfinite(rel_flow)
            is_infinite = ~is_finite
            
            if self._fun_reduce_n1 == "max":
                if is_infinite.any():
                    n1_vals[id_] = 5.  # some kind of infinite
                else:
                    n1_vals[id_] = rel_flow[is_finite].max()
            elif self._fun_reduce_n1 == "count":
                n1_vals[id_] = (rel_flow[is_finite] >= 1.).sum()
                n1_vals[id_] += (~is_finite).sum() 
            elif self._fun_reduce_n1 == "sum":
                n1_vals[id_] = rel_flow[is_finite].sum()
                n1_vals[id_] += (~is_finite).sum() * 5.
            else:
                raise RuntimeError("Unknown way to summarize information for each n1, found "
                                   f"'{self._fun_reduce_n1}', please use one of "
                                   "'max', 'count' or 'sum'")
            
        # summarize the n-1 information  for all n1 into 
        # one single reward score          
        if self._fun_reduce_reward == "max":
            return -float(n1_vals.max())
        if self._fun_reduce_reward == "count":
            res = (n1_vals >= 1.).sum()
            return -float(res)
        if self._fun_reduce_reward == "sum":
            res = n1_vals.sum()
            return -res
        raise RuntimeError("Unknown way to summarize n1 information, found "
                            f"'{self._fun_reduce_reward}', please use one of "
                            "'max', 'count' or 'sum'")
