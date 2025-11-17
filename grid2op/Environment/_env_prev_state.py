# Copyright (c) 2025, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import copy
from typing import Optional, Type, Union
import numpy as np
from grid2op.Space import GridObjects
import grid2op.Backend
from grid2op.typing_variables import CLS_AS_DICT_TYPING
from grid2op.Exceptions import Grid2OpException


class _EnvPreviousState(object):
    ERR_MSG_IMP_MODIF = "Impossible to modifiy this _EnvPreviousState"
    
    def __init__(self,
                 grid_obj_cls: Union[Type[GridObjects], CLS_AS_DICT_TYPING],
                 init_load_p : np.ndarray,
                 init_load_q : np.ndarray,
                 init_gen_p : np.ndarray,
                 init_gen_v : np.ndarray,
                 init_topo_vect : np.ndarray,
                 init_storage_p : np.ndarray,
                 init_shunt_p : np.ndarray,
                 init_shunt_q : np.ndarray,
                 init_shunt_bus : np.ndarray,
                 init_switch_state: Optional[np.ndarray]=None):
        self._can_modif = True
        if isinstance(grid_obj_cls, type):
            self._grid_obj_cls : CLS_AS_DICT_TYPING = grid_obj_cls.cls_to_dict()
        elif isinstance(grid_obj_cls, dict):
            self._grid_obj_cls : CLS_AS_DICT_TYPING = grid_obj_cls
        self._n_storage = len(self._grid_obj_cls["name_storage"])  # to avoid typing that over and over again
        
        self._load_p : np.ndarray = 1. * init_load_p
        self._load_q : np.ndarray = 1. * init_load_q
        self._gen_p : np.ndarray = 1. * init_gen_p
        self._gen_v : np.ndarray = 1. * init_gen_v
        self._storage_p : np.ndarray = 1. * init_storage_p
        self._topo_vect : np.ndarray = 1 * init_topo_vect
        self._shunt_p : np.ndarray = 1. * init_shunt_p
        self._shunt_q : np.ndarray = 1. * init_shunt_q
        self._shunt_bus : np.ndarray = 1. * init_shunt_bus
        if "detailed_topo_desc" in self._grid_obj_cls and self._grid_obj_cls["detailed_topo_desc"] is not None:
            self._switch_state = 1 * init_switch_state
        else:
            self._switch_state = None
            
    def copy(self):
        return _EnvPreviousState(grid_obj_cls=self._grid_obj_cls,
                                 init_load_p=self._load_p,
                                 init_load_q=self._load_q,
                                 init_gen_p=self._gen_p,
                                 init_gen_v=self._gen_v,
                                 init_topo_vect=self._topo_vect,
                                 init_storage_p=self._storage_p,
                                 init_shunt_p=self._shunt_p,
                                 init_shunt_q=self._shunt_q,
                                 init_shunt_bus = self._shunt_bus,
                                 init_switch_state=self._switch_state,
                                 )
        
    def update(self,
               load_p : np.ndarray,
               load_q : np.ndarray,
               gen_p : np.ndarray,
               gen_v : np.ndarray,
               topo_vect : np.ndarray,
               storage_p : Optional[np.ndarray],
               shunt_p : Optional[np.ndarray],
               shunt_q : Optional[np.ndarray],
               shunt_bus : Optional[np.ndarray],
               switches : Optional[np.ndarray],
               ):
        if not self._can_modif:
            raise Grid2OpException(type(self).ERR_MSG_IMP_MODIF)
        
        self._aux_update(topo_vect[self._grid_obj_cls["load_pos_topo_vect"]],
                         self._load_p,
                         load_p,
                         self._load_q,
                         load_q)
        self._aux_update(topo_vect[self._grid_obj_cls["gen_pos_topo_vect"]],
                         self._gen_p,
                         gen_p,
                         self._gen_v,
                         gen_v)
        self._topo_vect[topo_vect > 0] = 1 * topo_vect[topo_vect > 0]
        
        # update storage units
        if self._n_storage > 0:
            self._aux_update(topo_vect[self._grid_obj_cls["storage_pos_topo_vect"]],
                            self._storage_p,
                            storage_p)
        
        # handle shunts, if present
        if shunt_p is not None:
            self._aux_update(shunt_bus,
                             self._shunt_p,
                             shunt_p,
                             self._shunt_q,
                             shunt_q)
            self._shunt_bus[shunt_bus > 0] = 1 * shunt_bus[shunt_bus > 0]
            
        if switches is not None:
            if self._switch_state is None:
                raise Grid2OpException("No known last switch state to update")
            self._switch_state[:] = switches
        else:
            if self._switch_state is not None:
                raise Grid2OpException("No new switch values to update previous values")
                    
    def update_from_backend(self,
                            backend: "grid2op.Backend.Backend"):
        if not self._can_modif:
            raise Grid2OpException(type(self).ERR_MSG_IMP_MODIF)
        topo_vect = backend.get_topo_vect()
        load_p, load_q, *_ = backend.loads_info()
        gen_p, gen_q, gen_v = backend.generators_info()
        if self._n_storage > 0:
            storage_p, *_ = backend.storages_info()
        else:
            storage_p = None
        if type(backend).shunts_data_available:
            shunt_p, shunt_q, _, shunt_bus = backend.shunt_info()
        else:
            shunt_p, shunt_q, _, shunt_bus = None, None, None, None
            
        switches = None
        # if type(backend).detailed_topo_desc is not None:
        #     # TODO detailed topo !
        #     switches = np.ones(type(backend).detailed_topo_desc.switches.shape[0], dtype=dt_int)
        # else:
        #     switches = None
            
        self.update(load_p, load_q,
                    gen_p, gen_v,
                    topo_vect,
                    storage_p,
                    shunt_p, shunt_q, shunt_bus,
                    switches)
    
    def update_from_other(self, 
                          other : "_EnvPreviousState"):
        if not self._can_modif:
            raise Grid2OpException(type(self).ERR_MSG_IMP_MODIF)
        
        for attr_nm in ["_load_p",
                        "_load_q",
                        "_gen_p",
                        "_gen_v",
                        "_storage_p",
                        "_topo_vect",
                        "_shunt_p",
                        "_shunt_q",
                        "_shunt_bus"]:
            tmp = getattr(self, attr_nm)
            if tmp.size > 1:
                # works only for array of size 2 or more
                tmp[:] = copy.deepcopy(getattr(other, attr_nm))
            else:
                setattr(self, attr_nm, getattr(other, attr_nm))
        # if detailed topo
        if hasattr(self, "_switch_state") and self._switch_state is not None:
            self._switch_state[:] = other._switch_state
        
    def prevent_modification(self):
        self._aux_modif()
        self._can_modif = False
        
    def force_update(self, other: "_EnvPreviousState"):
        """This is used when initializing the forecast env. This removes the "cst" part, 
        set it to the value given by other, and then assign it to const.
        """
        self._can_modif = True
        self._aux_modif(True)
        self.update_from_other(other)
        self.prevent_modification()
    
    def _aux_modif(self, writeable_flag=False):
        for attr_nm in ["_load_p",
                        "_load_q",
                        "_gen_p",
                        "_gen_v",
                        "_storage_p",
                        "_topo_vect",
                        "_shunt_p",
                        "_shunt_q",
                        "_shunt_bus"]:
            tmp = getattr(self, attr_nm)
            if tmp.size > 1:
                # can't set flags on array of size 1 apparently
                tmp.flags.writeable = writeable_flag
                
        # if detailed topo
        if hasattr(self, "_switch_state") and self._switch_state is not None:
            self._switch_state.flags.writeable = writeable_flag
        
    def _aux_update(self,
                    el_topo_vect : np.ndarray,
                    arr1 : np.ndarray,
                    arr1_new : np.ndarray,
                    arr2 : Optional[np.ndarray] = None,
                    arr2_new : Optional[np.ndarray] = None):
        el_co = el_topo_vect > 0
        arr1[el_co] = 1. * arr1_new[el_co]
        if arr2 is not None:
            arr2[el_co] = 1. * arr2_new[el_co]
