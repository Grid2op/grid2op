# Copyright (c) 2025, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import importlib.metadata
from packaging import version

#: if not set by the user, in how many independant buses
#: each substation can be split. 2 by default
DEFAULT_N_BUSBAR_PER_SUB = 2

#: whether or not grid2op will allow (=continue an episode)
#: if a load a generator or a storage unit is disconnected
#: from the grid
DEFAULT_ALLOW_DETACHMENT = False

#: in which subfolder (of the environment) the grid2op classes
#: will be stored
GRID2OP_CLASSES_ENV_FOLDER = "_grid2op_classes"

#: current grid2op version, represented as a string
GRID2OP_CURRENT_VERSION_STR = importlib.metadata.version("grid2op")

#: current grid2op version used, represented as a "packaging version"
#: use this for any comparison
GRID2OP_CURRENT_VERSION = version.parse(GRID2OP_CURRENT_VERSION_STR)

#: TODO detailed topo
#: Legacy backend could not manipulate switches directly
DEFAULT_BK_CAN_MANIPULATE_SWITCH = False

#: legacy backend were required to manipulate
#: "bus breaker" / "nodal" topology
DEFAULT_BK_CAN_MANIPULATE_BUS_BREAKER = True