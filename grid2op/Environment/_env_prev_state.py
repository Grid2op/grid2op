# Copyright (c) 2025, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

from typing import Optional, Type, Any
import numpy as np
from grid2op.Space import GridObjects
import grid2op.Backend
from grid2op.typing_variables import CLS_AS_DICT_TYPING
from grid2op.Exceptions import Grid2OpException


class _EnvPreviousState(object):
    def __init__(self,
                 grid_obj_cls: Type[GridObjects],
                 init_load_p : np.ndarray,
                 init_load_q : np.ndarray,
                 init_gen_p : np.ndarray,
                 init_gen_v : np.ndarray,
                 init_topo_vect : np.ndarray,
                 init_storage_p : np.ndarray,
                 init_shunt_p : np.ndarray,
                 init_shunt_q : np.ndarray,
                 init_shunt_bus : np.ndarray):
        self._can_modif = True
        self._grid_obj_cls : CLS_AS_DICT_TYPING = grid_obj_cls.cls_to_dict()
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
        
    def update(self,
               load_p : np.ndarray,
               load_q : np.ndarray,
               gen_p : np.ndarray,
               gen_v : np.ndarray,
               topo_vect : np.ndarray,
               storage_p : Optional[np.ndarray],
               shunt_p : Optional[np.ndarray],
               shunt_q : Optional[np.ndarray],
               shunt_bus : Optional[np.ndarray]):
        if not self._can_modif:
            raise Grid2OpException(f"Impossible to modifiy this _EnvPreviousState")
        
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
    
    def update_from_backend(self,
                            backend: "grid2op.Backend.Backend"):
        if not self._can_modif:
            raise Grid2OpException(f"Impossible to modifiy this _EnvPreviousState")
        topo_vect = backend.get_topo_vect()
        load_p, load_q, *_ = backend.loads_info()
        gen_p, gen_q, gen_v = backend.generators_info()
        if self._n_storage > 0:
            storage_p, *_ = backend.storages_info()
        else:
            storage_p = None
        if type(backend).shunts_data_available:
            shunt_p, shunt_q, shunt_v, shunt_bus = backend.shunt_info()
        else:
            shunt_p, shunt_q, shunt_v, shunt_bus = None, None, None, None
        self.update(load_p, load_q,
                    gen_p, gen_v,
                    topo_vect,
                    storage_p,
                    shunt_p, shunt_q, shunt_bus)
    
    def update_from_other(self, 
                          other : "_EnvPreviousState"):
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
                tmp[:] = getattr(other, attr_nm)
            else:
                setattr(self, attr_nm, getattr(other, attr_nm))
        
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
        