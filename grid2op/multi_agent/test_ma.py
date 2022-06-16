# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.
import sys
import pathlib
filepath = pathlib.Path(__file__).resolve().parent.parent.parent
print(filepath)
print()
sys.path.insert(0, str(filepath))
print(sys.path)
import unittest
import warnings
from grid2op import make
from grid2op.multi_agent.multiAgentEnv import MultiAgentEnv
import pdb
import numpy as np
from grid2op.multi_agent.multi_agentExceptions import *



class MATester(unittest.TestCase):
    def setUp(self) -> None:
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = make("l2rpn_case14_sandbox", test = True)

        self.action_domains = {
            'agent_0' : [0,1,2,3, 4],
            'agent_1' : [5,6,7,8,9,10,11,12,13]
        }
        self.observation_domains = {
            'agent_0' : self.action_domains['agent_1'],
            'agent_1' : self.action_domains['agent_0']
        }
        # run redispatch agent on one scenario for 100 timesteps
        self.ma_env = MultiAgentEnv(self.env, self.observation_domains, self.action_domains)
        return super().setUp()
    
    def test_verify_domains(self):
        action_domains = {
            'agent_0' : 0,
            'agent_1' : [5,6,7,8,9,10,11,12,13]
        }
        observation_domains = action_domains
        try:
            MultiAgentEnv(self.env, observation_domains, action_domains)
            
        except DomainException :
            assert True
            return
            
        assert False
    
    # TODO Test in case subs are not connex

    # TODO 
    
    def test_build_subgrids_action_domains(self):
        """Tests that the action_domains are correctly defined 
            in MultiAgentEnv._build_subgrid_from_domain method
        """
        assert self.ma_env._action_domains['agent_0']['sub_id'] == self.action_domains['agent_0']
        assert self.ma_env._action_domains['agent_1']['sub_id'] == self.action_domains['agent_1']
        
        assert (self.ma_env._action_domains['agent_0']['mask_load'] == [True,  True,  True,  True, False, False, False, False, False, False, False]).all()
        assert (self.ma_env._action_domains['agent_1']['mask_load'] == np.invert([True,  True,  True,  True, False, False, False, False, False, False, False])).all()
        
        assert (self.ma_env._action_domains['agent_0']['mask_gen'] == [ True,  True, False, False, False,  True]).all()
        assert (self.ma_env._action_domains['agent_1']['mask_gen'] == np.invert([ True,  True, False, False, False,  True])).all()
        
        assert (self.ma_env._action_domains['agent_0']['mask_storage'] == []).all()
        assert (self.ma_env._action_domains['agent_1']['mask_storage'] == []).all()
        
        assert (self.ma_env._action_domains['agent_0']['mask_line_ex'] == [ True,  True,  True,  True,  True,  True,  True, False, False,False, False, False, False, False, False, False, False, False,False, False]).all()
        assert (self.ma_env._action_domains['agent_1']['mask_line_ex'] == np.invert([ True,  True,  True,  True,  True,  True,  True, False, False,False, False, False, False, False, False, True, True, True,False, False])).all()
        
        assert (self.ma_env._action_domains['agent_0']['mask_line_or'] == self.ma_env._action_domains['agent_0']['mask_line_ex']).all()
        assert (self.ma_env._action_domains['agent_1']['mask_line_or'] == 
                np.invert(self.ma_env._action_domains['agent_0']['mask_line_ex'])
                & self.ma_env._action_domains['agent_1']['mask_line_ex']
        ).all()
        
        assert (self.ma_env._action_domains['agent_0']['mask_shunt'] == [False]).all()
        assert (self.ma_env._action_domains['agent_1']['mask_shunt'] == [True]).all()
        
        assert (self.ma_env._action_domains['agent_0']['mask_interco'] == [False, False, False, False, False, False, False, False, False,
                                                                            False, False, False, False, False, False,  True,  True,  True,
                                                                            False, False]).all()
        assert (self.ma_env._action_domains['agent_1']['mask_interco'] == [False, False, False, False, False, False, False, False, False,
                                                                            False, False, False, False, False, False,  True,  True,  True,
                                                                            False, False]).all()
        
        assert (self.ma_env._action_domains['agent_0']['interco_is_origin'] == [True, True, True]).all()
        assert (self.ma_env._action_domains['agent_1']['interco_is_origin'] == np.invert([True, True, True])).all()
        
        
    #def test_build_subgrids_observation_domains(self):
    #    """Tests that the observation_domains are correctly defined 
    #        in MultiAgentEnv._build_subgrid_from_domain method
    #    """
    #    assert self.ma_env._observation_domains['agent_1']['sub_id'] == self.action_domains['agent_0']
    #    assert self.ma_env._observation_domains['agent_0']['sub_id'] == self.action_domains['agent_1']
    #    
    #    assert (self.ma_env._observation_domains['agent_1']['mask_load'] == [True,  True,  True,  True, False, False, False, False, False, False, False]).all()
    #    assert (self.ma_env._observation_domains['agent_0']['mask_load'] == np.invert([True,  True,  True,  True, False, False, False, False, False, False, False])).all()
    #    
    #    assert (self.ma_env._observation_domains['agent_1']['mask_gen'] == [True,  True, False, False, False,  True]).all()
    #    assert (self.ma_env._observation_domains['agent_0']['mask_gen'] == np.invert([ True,  True, False, False, False,  True])).all()
    #    
    #    assert (self.ma_env._observation_domains['agent_1']['mask_storage'] == []).all()
    #    assert (self.ma_env._observation_domains['agent_0']['mask_storage'] == []).all()
    #    
    #    assert (self.ma_env._observation_domains['agent_1']['mask_line_ex'] == [ True,  True,  True,  True,  True,  True,  True, False, False, False, False, False, False, False, False, False, False, False,False, False]).all()
    #    assert (self.ma_env._observation_domains['agent_0']['mask_line_ex'] == np.invert([ True,  True,  True,  True,  True,  True,  True, False, False, False, False, False, False, False, False, False, False, False,False, False])).all()
    #    
    #    assert (self.ma_env._observation_domains['agent_1']['mask_line_or'] == self.ma_env._observation_domains['agent_1']['mask_line_ex'] ).all()
    #    assert (self.ma_env._observation_domains['agent_0']['mask_line_or'] == np.invert(self.ma_env._observation_domains['agent_1']['mask_line_ex'])).all()
    #    
    #    assert (self.ma_env._observation_domains['agent_1']['mask_shunt'] == [False]).all()
    #    assert (self.ma_env._observation_domains['agent_0']['mask_shunt'] == [True]).all()
    #    
    #    assert (self.ma_env._observation_domains['agent_0']['mask_interco'] == [False, False, False, False, False, False, False, False, False,
    #                                                                        False, False, False, False, False, False,  True,  True,  True,
    #                                                                        False, False]).all()
    #    assert (self.ma_env._observation_domains['agent_1']['mask_interco'] == [False, False, False, False, False, False, False, False, False,
    #                                                                        False, False, False, False, False, False,  True,  True,  True,
    #                                                                        False, False]).all()
    #    
    #    assert (self.ma_env._observation_domains['agent_1']['interco_is_origin'] == [True, True, True]).all()
    #    assert (self.ma_env._observation_domains['agent_0']['interco_is_origin'] == np.invert([True, True, True])).all()
    
    # TODO
    def test_validate_action_domain(self):
        """test the MultiAgentEnv._verify_domains method """
        # TODO it should test that :
        # 1) the function works (does not throw an error) when the input domains are correct
        # 2) the function throws an error when the input domains are wrong
        # (the more "wrong" cases tested the better)
        pass

    def test_build_subgrid_obj(self):
        """test the MultiAgentEnv._build_subgrid_obj_from_domain"""
        # TODO test that this function creates an object with the right
        # attributes and the right values from the action / observation
        # domain
        pass
    
    def test_action_space(self):
        """test for the action spaces created for agents
        """
        try:
            #Simple do nothing action
            print(self.ma_env.action_spaces['agent_0']({}))
            assert True
        except Exception as e:
            assert False
            
        try:
            #action on a line
            print(self.ma_env.action_spaces['agent_0']({
                'change_bus' : self.ma_env.action_spaces['agent_0'].line_or_pos_topo_vect[0]
            }))
            print(self.ma_env.action_spaces['agent_1']({
                'change_bus' : self.ma_env.action_spaces['agent_0'].line_ex_pos_topo_vect[0]
            }))
            assert True
        except Exception as e:
            assert False
            
        try:
            #action on a gen
            print(self.ma_env.action_spaces['agent_0']({
                'change_bus' : self.ma_env.action_spaces['agent_0'].gen_pos_topo_vect[0]
            }))
            assert True
        except Exception as e:
            assert False
            
        try:
            #action on a load
            print(self.ma_env.action_spaces['agent_0']({
                'change_bus' : self.ma_env.action_spaces['agent_0'].load_pos_topo_vect[0]
            }))
            assert True
        except Exception as e:
            assert False
        
        try:
            #action on an interconnection
            print(self.ma_env.action_spaces['agent_0']({
                'change_bus' : self.ma_env.action_spaces['agent_0'].interco_pos_topo_vect[0]
            }))
            assert True
        except Exception as e:
            assert False
    
if __name__ == "__main__":
    unittest.main()
