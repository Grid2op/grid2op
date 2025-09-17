# Copyright (c) 2025, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.
import os

import sys
if sys.version_info >= (3, 9):
    import importlib.resources as importlib_resources
else:
    import importlib_resources
    
    
DEV_DATA_FOLDER = importlib_resources.files("grid2op") / "data"
DEV_DATASET = os.path.join(DEV_DATA_FOLDER, "{}")
TEST_DEV_ENVS = {
    "blank": DEV_DATASET.format("blank"),
    "rte_case14_realistic": DEV_DATASET.format("rte_case14_realistic"),
    "rte_case14_redisp": DEV_DATASET.format("rte_case14_redisp"),
    "rte_case14_test": DEV_DATASET.format("rte_case14_test"),
    "rte_case5_example": DEV_DATASET.format("rte_case5_example"),
    "rte_case5_flexibility": DEV_DATASET.format("rte_case5_flexibility"),
    "rte_case118_example": DEV_DATASET.format("rte_case118_example"),
    "rte_case14_opponent": DEV_DATASET.format("rte_case14_opponent"),
    "l2rpn_wcci_2020": DEV_DATASET.format("l2rpn_wcci_2020"),
    "l2rpn_neurips_2020_track2": DEV_DATASET.format("l2rpn_neurips_2020_track2"),
    "l2rpn_neurips_2020_track1": DEV_DATASET.format("l2rpn_neurips_2020_track1"),
    "l2rpn_case14_sandbox": DEV_DATASET.format("l2rpn_case14_sandbox"),
    "l2rpn_case14_sandbox_diff_grid": DEV_DATASET.format("l2rpn_case14_sandbox_diff_grid"),
    "l2rpn_icaps_2021": DEV_DATASET.format("l2rpn_icaps_2021"),
    "l2rpn_wcci_2022_dev": DEV_DATASET.format("l2rpn_wcci_2022_dev"),
    "l2rpn_wcci_2022": DEV_DATASET.format("l2rpn_wcci_2022_dev"),
    "l2rpn_idf_2023": DEV_DATASET.format("l2rpn_idf_2023"),
    # educational files
    "educ_case14_redisp": DEV_DATASET.format("educ_case14_redisp"),
    "educ_case14_storage": DEV_DATASET.format("educ_case14_storage"),
    # keep the old names for now
    "case14_realistic": DEV_DATASET.format("rte_case14_realistic"),
    "case14_redisp": DEV_DATASET.format("rte_case14_redisp"),
    "case14_test": DEV_DATASET.format("rte_case14_test"),
    "case5_example": DEV_DATASET.format("rte_case5_example"),
    "case14_fromfile": DEV_DATASET.format("rte_case14_test"),
}