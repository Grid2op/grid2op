# Copyright (c) 2024, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import copy
import numpy as np
from os import PathLike
from typing import Union
import grid2op
from grid2op.Backend import PandaPowerBackend
import unittest
import warnings

import pdb

from grid2op.Backend.pandaPowerBackend import NUMBA_
from grid2op.Action._backendAction import _BackendAction


class _Aux_Test_PPDiffOrder(PandaPowerBackend):
    def __init__(self,
                 detailed_infos_for_cascading_failures: bool = False,
                 lightsim2grid: bool = False,
                 dist_slack: bool = False,
                 max_iter: int = 10,
                 can_be_copied: bool = True,
                 with_numba: bool = NUMBA_,
                 seed=0):
        super().__init__(detailed_infos_for_cascading_failures,
                         lightsim2grid,
                         dist_slack,
                         max_iter,
                         can_be_copied,
                         with_numba)
        self._order_line = None
        self._order_load = None
        self._inv_order_line = None
        self._inv_order_load = None
        self.seed = seed
        self._prng = np.random.default_rng(seed)
        self.li_attr_del = ["gen_to_sub_pos",
                            "load_to_sub_pos",
                            "line_or_to_sub_pos",
                            "line_ex_to_sub_pos"
                            ]
        self.li_pos_topo_vect = ["line_or_pos_topo_vect",
                                 "line_ex_pos_topo_vect",
                                 "load_pos_topo_vect",
                                 "gen_pos_topo_vect",
                                 ]
        self._orig_topo_vect = None
        self._new_topo_vect = None
        
        self._my_kwargs["seed"] = int(self.seed)
        
    def load_grid(self, path: Union[PathLike, str], filename: Union[PathLike, str, None] = None) -> None:
        super().load_grid(path, filename)
        if self.n_storage > 0:
            self.li_attr_del.append("storage_to_sub_pos")
            self.li_pos_topo_vect.append("storage_pos_topo_vect")
        
        self._orig_topo_vect = {el: getattr(type(self), el) for el in self.li_pos_topo_vect}
        
        # generate a different order
        self._order_line = np.arange(self.n_line)
        self._prng.shuffle(self._order_line)
        self._order_load = np.arange(self.n_load)
        self._prng.shuffle(self._order_load)
        self._inv_order_load = np.argsort(self._order_load)
        self._inv_order_line = np.argsort(self._order_line)
        
        # load the grid
        self.load_to_subid = self.load_to_subid[self._order_load]
        self.line_or_to_subid = self.line_or_to_subid[self._order_line]
        self.line_ex_to_subid = self.line_ex_to_subid[self._order_line]
        
        # delete all the set attribute by the PandaPowerBackend class
        for attr_nm in self.li_attr_del:
            delattr(self, attr_nm)
            setattr(self, attr_nm, None)
        
        # compute the "big topo" position
        self._compute_pos_big_topo() 
        self.thermal_limit_a = self.thermal_limit_a[self._order_line]
        self._new_topo_vect = {el: getattr(type(self), el) for el in self.li_pos_topo_vect}
        self.name_load = self.name_load[self._order_load]
        self.name_line = self.name_line[self._order_line]
        
        self._init_bus_load = self._init_bus_load[self._order_load]
        self._init_bus_lor = self._init_bus_lor[self._order_line]
        self._init_bus_lex = self._init_bus_lex[self._order_line]
        self._init_big_topo_to_bk()
        self._init_topoid_objid()
    
    def apply_action(self, backendAction: _BackendAction) -> None:
        if backendAction is None:
            return
        reordered = copy.deepcopy(backendAction)
        reordered.load_p.reorder(self._inv_order_load)
        reordered.load_q.reorder(self._inv_order_load)
        # probably won't work if topo is changed...
        return super().apply_action(reordered)
    
    def _loads_info(self):
        tmp = super()._loads_info()
        res = [el[self._order_load] for el in tmp]
        return res
    
    def _aux_get_line_info(self, colname1, colname2):
        vect = super()._aux_get_line_info(colname1, colname2)
        return vect[self._order_line]
    
    def get_class_added_name(self) -> str:
        return type(self).__name__ + f"_{self.seed}"
    
class TestAddClassNameBackend(unittest.TestCase):
    def setUp(self) -> None:
        self.tgt_load_p = np.array( [22.        , 87.        , 45.79999924,  7.        , 12.        ,
                                     28.20000076,  8.69999981,  3.5       ,  5.5       , 12.69999981,
                                     14.80000019])
        self.load_pos_topo_vect_diff_order = np.array([13, 44, 19, 41, 54, 36, 24,  9,  3, 47, 50])
        self.line_or_pos_topo_vect_diff_order = np.array([ 5, 37, 14,  6, 48, 15,  7, 38, 39, 27,  
                                                           1, 42, 28, 11, 31, 20, 51, 29,  2, 16])
        self.load_pos_topo_vect_corr_order = np.array([ 8, 12, 18, 23, 30, 40, 43, 46, 49, 53, 56])
        self.line_or_pos_topo_vect_corr_order = np.array([ 0,  1,  4,  5,  6, 10, 15, 24, 25, 26, 
                                                          36, 37, 42, 48, 52, 16, 17, 22, 32, 39])
        
        self.load_pos_topo_vect_multi_do = np.array([ 23, 118, 165, 200, 364, 512,  76, 495, 429, 121,  35, 522, 174,
                                                     203, 281, 389, 271, 377,  95,  89, 181, 447, 100, 298, 187, 432,
                                                     450, 530, 484, 411, 184, 502, 246,  92, 241, 259, 230, 361, 220,
                                                     491,   0, 453, 474, 141, 344, 330,  42, 456, 519,  54, 420, 386,
                                                     471, 338, 256, 335, 132, 401,  86,   3,  66, 223, 150, 196, 227,
                                                      80,  26, 305, 468, 138, 348, 515, 262, 319, 505,  57, 381,  69,
                                                     333, 525, 479,  20, 162, 233, 128, 396,   6, 499, 417, 358, 171,
                                                     438,  10, 191, 147, 528, 111, 441,  51])
        self.load_pos_topo_vect_multi_pp = np.array([  2,   5,   9,  14,  22,  25,  30,  41,  50,  53,  56,  65,  68,
                                                      75,  79,  85,  88,  91,  94,  99, 103, 117, 120, 123, 131, 137,
                                                     140, 146, 149, 152, 164, 170, 173, 180, 183, 186, 190, 195, 199,
                                                     202, 219, 222, 226, 229, 232, 240, 245, 255, 258, 261, 270, 275,
                                                     287, 304, 307, 326, 332, 334, 337, 343, 347, 357, 360, 363, 374,
                                                     380, 385, 388, 395, 398, 403, 416, 419, 428, 431, 437, 440, 446,
                                                     449, 452, 455, 467, 470, 473, 478, 483, 490, 494, 498, 501, 504,
                                                     509, 514, 518, 521, 524, 527, 529, 532])
        return super().setUp()
    
    def get_env_name(self):
        return "l2rpn_case14_sandbox"
    
    def get_env_name_multi(self):
        return "l2rpn_neurips_2020_track2"
    
    def debug_fake_backend(self):
        tgt_load_bus = np.array([ 1,  2,  3,  4,  5,  8,  9, 10, 11, 12, 13])
        env1 = grid2op.make(self.get_env_name(), test=True, backend=_Aux_Test_PPDiffOrder(seed=0), _add_cls_nm_bk=False, _add_to_name=type(self).__name__)    
        assert (env1.load_pos_topo_vect == self.load_pos_topo_vect_diff_order ).all()
        assert (env1.line_or_pos_topo_vect == self.line_or_pos_topo_vect_diff_order).all()
        env1.reset(seed=0, options={"time serie id": 0})
        assert np.abs(env1.backend._grid.load["p_mw"] - self.tgt_load_p).max() <= 1e-5
        assert np.all(env1.backend._grid.load["bus"] == tgt_load_bus)
    
    def test_legacy_behaviour_fails(self):
        test_id = "0"
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env0_0 = grid2op.make(self.get_env_name(), test=True, _add_cls_nm_bk=False, _add_to_name=type(self).__name__+test_id)  
            env0_1 = grid2op.make(self.get_env_name(), test=True, backend=_Aux_Test_PPDiffOrder(seed=0), _add_cls_nm_bk=False, _add_to_name=type(self).__name__+test_id)   
        assert type(env0_0).__name__ == type(env0_1).__name__
        assert (env0_0.load_pos_topo_vect == self.load_pos_topo_vect_corr_order ).all()
        assert (env0_0.line_or_pos_topo_vect == self.line_or_pos_topo_vect_corr_order).all()
        assert (env0_1.load_pos_topo_vect != self.load_pos_topo_vect_diff_order ).any()
        assert (env0_1.line_or_pos_topo_vect != self.line_or_pos_topo_vect_diff_order).any()
        
        test_id = "1"
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env1_0 = grid2op.make(self.get_env_name(), test=True, backend=_Aux_Test_PPDiffOrder(seed=0), _add_cls_nm_bk=False, _add_to_name=type(self).__name__+test_id)   
            env1_1 = grid2op.make(self.get_env_name(), test=True, _add_cls_nm_bk=False, _add_to_name=type(self).__name__+test_id)  
        assert type(env1_0).__name__ == type(env1_1).__name__
        assert (env1_0.load_pos_topo_vect == self.load_pos_topo_vect_diff_order ).all()
        assert (env1_0.line_or_pos_topo_vect == self.line_or_pos_topo_vect_diff_order).all()
        assert (env1_1.load_pos_topo_vect != self.load_pos_topo_vect_corr_order ).any()
        assert (env1_1.line_or_pos_topo_vect != self.line_or_pos_topo_vect_corr_order).any()
        
    def test_basic_env(self):
        test_id = "3"
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env0 = grid2op.make(self.get_env_name(), test=True, backend=_Aux_Test_PPDiffOrder(seed=0), _add_to_name=type(self).__name__+test_id)   
            env1 = grid2op.make(self.get_env_name(), test=True, _add_to_name=type(self).__name__+test_id)  
        assert type(env0).__name__ != type(env1).__name__
        assert (env0.load_pos_topo_vect == self.load_pos_topo_vect_diff_order ).all()
        assert (env0.line_or_pos_topo_vect == self.line_or_pos_topo_vect_diff_order).all()
        assert (env1.load_pos_topo_vect == self.load_pos_topo_vect_corr_order ).all()
        assert (env1.line_or_pos_topo_vect == self.line_or_pos_topo_vect_corr_order).all()
        
        test_id = "4"
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env0 = grid2op.make(self.get_env_name(), test=True, _add_to_name=type(self).__name__+test_id)  
            env1 = grid2op.make(self.get_env_name(), test=True, backend=_Aux_Test_PPDiffOrder(seed=0), _add_to_name=type(self).__name__+test_id)   
        assert type(env0).__name__ != type(env1).__name__
        assert (env1.load_pos_topo_vect == self.load_pos_topo_vect_diff_order ).all()
        assert (env1.line_or_pos_topo_vect == self.line_or_pos_topo_vect_diff_order).all()
        assert (env0.load_pos_topo_vect == self.load_pos_topo_vect_corr_order ).all()
        assert (env0.line_or_pos_topo_vect == self.line_or_pos_topo_vect_corr_order).all()
    
    def test_multi_env(self):
        test_id = "5"
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env0 = grid2op.make(self.get_env_name_multi(), test=True, backend=_Aux_Test_PPDiffOrder(seed=0), _add_to_name=type(self).__name__+test_id)   
            env1 = grid2op.make(self.get_env_name_multi(), test=True, _add_to_name=type(self).__name__+test_id)  
        assert (type(env0).load_pos_topo_vect == self.load_pos_topo_vect_multi_do).all()
        for el in env0:
            assert (type(el).load_pos_topo_vect == self.load_pos_topo_vect_multi_do).all()
        assert (type(env1).load_pos_topo_vect == self.load_pos_topo_vect_multi_pp).all()
        for el in env1:
            assert (type(el).load_pos_topo_vect == self.load_pos_topo_vect_multi_pp).all()
            
        test_id = "6"
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore") 
            env0 = grid2op.make(self.get_env_name_multi(), test=True, _add_to_name=type(self).__name__+test_id)  
            env1 = grid2op.make(self.get_env_name_multi(), test=True, backend=_Aux_Test_PPDiffOrder(seed=0), _add_to_name=type(self).__name__+test_id)  
        assert (type(env1).load_pos_topo_vect == self.load_pos_topo_vect_multi_do).all()
        for el in env1:
            assert (type(el).load_pos_topo_vect == self.load_pos_topo_vect_multi_do).all()
        assert (type(env0).load_pos_topo_vect == self.load_pos_topo_vect_multi_pp).all()
        for el in env0:
            assert (type(el).load_pos_topo_vect == self.load_pos_topo_vect_multi_pp).all()
        
# TODO and as always, add Runner, MaskedEnv and TimedOutEnv
# TODO check with "automatic class generation"