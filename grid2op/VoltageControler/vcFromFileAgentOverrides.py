# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import numpy as np
from typing import TYPE_CHECKING

from grid2op.VoltageControler.ControlVoltageFromFile import ControlVoltageFromFile

if TYPE_CHECKING:
    from grid2op.Observation import BaseObservation
    from grid2op.Action import BaseAction


class VCFromFileAgentOverrides(ControlVoltageFromFile):
    """
    
    .. versionadded: 1.12.2
    
    This class allows the agent to override the voltage setpoint 
    of the generators.
    
    It allows the agent to modifies the generator voltages. If a generator
    is not modified by the agent, then the value (for this generator) is
    read from the time series (same behaviour for this generator as 
    the default :class:`grid2op.VoltageControler.ControlVoltageFromFile` class.)
    
    To use it, you can do:
        
    .. code-block:: python
    
        import grid2op
        from grid2op.VoltageControler import VCFromFileAgentOverrides
        
        env_name = ...
        
        env = grid2op.make(env_name,
                            ..., 
                            voltagecontroler_class=VCFromFileAgentOverrides)
                            
    And if you want your agent to perform action on generator setpoint you can 
    specify it like this.
    
    If you want to modify all the generator at the same time:
    
    .. code-block:: python

        obs = env.reset()
        
        new_v_setpoint = 1. * obs.gen_v
        
        # specify the new setpoints
        new_v_setpoint *= 1.01  # increase it by 1%, for example...
        #######
        
        modify_prod_v_value = env.action_space({"injection": {"prod_v": new_prod_v}})
        
        obs, reward, done, info = env.step(modify_prod_v_value)
        print(obs.gen_v)
        
    If you want to modify the setpoint of only one generator:
    
    
    .. code-block:: python

        obs = env.reset()
        
        new_v_setpoint = 1. * obs.gen_v
        
        # specify the new setpoints
        # for example say we want to tell the generator 0 
        # to have a setpoint of 145 kV

        new_v_setpoint[0] = 145.
        #######
        
        modify_prod_v_value = env.action_space({"injection": {"prod_v": new_prod_v}})
        
        obs, reward, done, info = env.step(modify_prod_v_value)
        print(obs.gen_v)
        
    """
    def fix_voltage(self,
                    observation: "BaseObservation",
                    agent_action: "BaseAction",
                    env_action: "BaseAction",
                    prod_v_chronics: np.ndarray):
        vect_ = None
        if "prod_v" in agent_action._dict_inj and np.isfinite(agent_action._dict_inj["prod_v"]).any():
            # agent decision
            vect_ = 1. * agent_action._dict_inj["prod_v"]
        if prod_v_chronics is not None:
            # default values (in the time series)
            if vect_ is None:
                # agent did not change anything
                vect_ = prod_v_chronics
            else:
                # keep the agent choice and put the default values
                # (from the time series) for the generator not modified
                # by the agent
                mask_default = ~np.isfinite(vect_)
                vect_[mask_default] = prod_v_chronics[mask_default]
                
        # now build the action
        if vect_ is not None:
            res = self.action_space({"injection": {"prod_v": vect_}})
        else:
            res = self.action_space()
            
        if observation is not None:
            # cache the get_topological_impact to avoid useless computations later
            # this is a speed optimization
            _ = res.get_topological_impact(observation.line_status, _store_in_cache=True, _read_from_cache=False)
            return res
        return super().fix_voltage(observation, agent_action, env_action, prod_v_chronics)
