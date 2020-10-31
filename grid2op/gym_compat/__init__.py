# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.
__all__ = ["BaseGymAttrConverter", "GymEnv"]

from grid2op.gym_compat.base_gym_attr_converter import BaseGymAttrConverter
from grid2op.gym_compat.gymenv import GymEnv

try:
    from grid2op.gym_compat.gym_space_converter import GymObservationSpace, GymActionSpace
    __all__.append("GymObservationSpace")
    __all__.append("GymActionSpace")
except ImportError:
    # you must install open ai gym to benefit from this converter
    pass

try:
    from grid2op.gym_compat.scaler_attr_converter import ScalerAttrConverter
    __all__.append("ScalerAttrConverter")
except ImportError:
    # you must install open ai gym to benefit from this converter
    pass

try:
    from grid2op.gym_compat.multi_to_tuple_converter import MultiToTupleConverter
    __all__.append("MultiToTupleConverter")
except ImportError:
    # you must install open ai gym to benefit from this converter
    pass