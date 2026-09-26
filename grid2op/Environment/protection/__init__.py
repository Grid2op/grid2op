# Copyright (c) 2026, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

__all__ = [
    "Protection",
    "ProtectionConfig",
    "ProtectionState",
    "default_from_parameters",
    "PROTECTIONS_FILE_NAME",
]

from grid2op.Environment.protection.protection import Protection, ProtectionConfig, default_from_parameters
from grid2op.Environment.protection.protection_state import ProtectionState

#: name of the file, in the environment directory, that defines the protections
PROTECTIONS_FILE_NAME = "protections.json"
