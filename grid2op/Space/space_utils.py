# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import copy
from typing import Optional
import numpy as np

from grid2op.Exceptions import Grid2OpException

# i already issued the warning for the "some substations have no controllable elements"
_WARNING_ISSUED_FOR_SUB_NO_ELEM = False
# this global variable is not const ! It is modified in GridObjects.py


def extract_from_dict(dict_, key, converter):
    if key not in dict_:
        raise Grid2OpException(
            'Impossible to find key "{}" while loading the dictionary.'.format(key)
        )
    try:
        res = converter(dict_[key])
    except Exception as exc_:
        raise Grid2OpException(
            'Impossible to convert "{}" into class {} with exception '
            '\n"{}"'.format(key, converter, exc_)
        )
    return res


def save_to_dict(res_dict, me, key, converter, copy_=True):
    """

    Parameters
    ----------
    res_dict:
        output dictionary
    me:
        the object to serialize in a dict
    key:
        the attribute of the object we want to save
    converter:
        if the attribute need to be converted (for example if you later want to serialize the dictionary as json)
    copy_:
        whether you copy the attribute or not (only applies if converter is None)

    Returns
    -------

    """
    if not hasattr(me, key):
        raise Grid2OpException(
            'Impossible to find key "{}" while loading the dictionary.'.format(key)
        )
    try:
        if converter is not None:
            res = converter(getattr(me, key))
        else:
            if copy_:
                res = copy.deepcopy(getattr(me, key))
            else:
                res = getattr(me, key)
    except Exception as exc_:
        raise Grid2OpException(
            'Impossible to convert "{}" into class {} with exception '
            '\n"{}"'.format(key, converter, exc_)
        )

    if key in res_dict:
        msg_err_ = (
            'Key "{}" is already present in the result dictionary. This would override it'
            " and is not supported."
        )
        raise Grid2OpException(msg_err_.format(key))
    res_dict[key] = res


class ElTypeInfo:
    """
    For each element type (*eg* generator, or load) this is a container for basic element
    information, it is used in the `check_kirchhoff` functions  of observation :func:`grid2op.Observation.BaseObservation.check_kirchhoff`
    and backend :func:`grid2op.Backend.Backend.check_kirchhoff` 
    
    The element it contains are:
    
        - el_bus : np.ndarray
          for each element of this element type, on which busbar this element is connected (*eg* obs.gen_bus)
        - el_p : np.ndarray
          for each element of this element type, it gives the active power value consumed / produced by this element (*eg* obs.gen_p)
        - el_q : np.ndarray
          for each element of this element type, it gives the reactive power value consumed / produced by this element (*eg* obs.gen_q)
        - el_v : np.ndarray
          for each element of this element type, it gives the voltage magnitude of the bus to which this element (*eg* obs.gen_v)
          is connected
          
    """
    def __init__(self,
                 el_bus : np.ndarray, 
                 el_p : np.ndarray,
                 el_q : np.ndarray, 
                 el_v : Optional[np.ndarray] = None,
                #  load_conv: bool = True
                 ):
        self._bus = el_bus
        self._p = el_p
        self._q = el_q
        self._v = el_v
