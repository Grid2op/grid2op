# Copyright (c) 2025, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.
import json
from abc import ABC
from datetime import datetime
from pathlib import Path
from typing import Tuple, Union, Callable, List

import pyarrow as pa
import pyarrow.parquet

from grid2op.Action import BaseAction
from grid2op.Environment.EnvInterface import EnvInterface
from grid2op.Observation import BaseObservation
from grid2op.typing_variables import STEP_INFO_TYPING, RESET_OPTIONS_TYPING


class AbstractTable(ABC):
    """
    A class to accumulate, organize, and write objects data in a columnar format
    to Parquet files. Designed to handle large-scale data efficiently, buffering
    objects before writing in chunks.

    This class is intended to facilitate data management for time-stamped objects,
    appending new objects vectors, and exporting them to disk efficiently to
    reduce memory usage and improve disk I/O performance over time.

    Attributes
    ----------
    _columns : List[str]
        List of column names representing the structure of the object vector.

    _directory : Path
        Path to the directory where the Parquet file will be stored.

    _table_name : str
        Name of the output Parquet file (without the extension).

    _write_chunk_size : int
        Number of rows to buffer before writing to the Parquet file.

    _buffer : List[List]
        Internal buffer to temporarily store observation data before writing.

    _writer : Optional[pa.parquet.ParquetWriter]
        Writer object to manage Parquet file I/O operations, lazy initialized.

    """
    def __init__(self, columns: List[str], directory: Path, table_name: str, write_chunk_size: int):
        self._columns = columns
        self._directory = directory
        self._table_name = table_name
        self._write_chunk_size = write_chunk_size
        self._buffer = [[] for _ in range(len(columns) + 1)]
        self._writer = None

    def reset(self):
        self.close() # or with discard buffered data ?

    def _flush(self, force: bool):
        if force or len(self._buffer[0]) >= self._write_chunk_size:
            table = pa.table(self._buffer, ['time'] + list(self._columns))
            if self._writer is None:
                parquet_file = self._directory / f"{self._table_name}.parquet"
                self._writer = pa.parquet.ParquetWriter(parquet_file, schema=table.schema)
            self._writer.write_table(table)
            self._buffer = [[] for _ in range(len(self._columns) + 1)] # reset buffer

    def close(self):
        self._flush(True)
        if self._writer is not None:
            self._writer.close()
            self._writer = None


ObservationVectorGetter = Callable[[BaseObservation], List[float]]

class ObservationTable(AbstractTable):

    def __init__(self, columns: List[str], getter: ObservationVectorGetter, directory: Path, table_name: str,
                 write_chunk_size: int):
        super().__init__(columns, directory, table_name, write_chunk_size)
        self._getter = getter

    def append(self, obs: BaseObservation):
        time = obs.get_time_stamp()
        self._buffer[0].append(int(time.timestamp()))

        vec = self._getter(obs)
        for i in range(len(self._columns)):
            self._buffer[i + 1].append(vec[i])

        self._flush(False)


class ActionTable(AbstractTable):

    def __init__(self, directory: Path, table_name: str, write_chunk_size: int):
        super().__init__(['action', 'done'], directory, table_name, write_chunk_size)

    def append(self, time: datetime, act: BaseAction, done: bool):
        self._buffer[0].append(int(time.timestamp()))
        json_str = json.dumps(act.as_serializable_dict())
        self._buffer[1].append(json_str)
        self._buffer[2].append(done)
        self._flush(False)


class EnvRecorder(EnvInterface):
    """
    An environment recorder for capturing and storing environment data.

    This class serves as a wrapper for a given environment and records its
    observations into Parquet files for later analysis. It ensures that environment
    data such as observations are properly stored in a structured format.

    Attributes
    ----------

    _env : EnvInterface
        The underlying environment to be wrapped and recorded.

    _tables : list of ObservationTable
        A list of observation tables used to record specific environment
        observations, such as generator power or load power.

    """
    def __init__(self, env, directory: Path, write_chunk_size: int = 1000):
        super().__init__()
        self._env = env
        self._directory = directory

        # save the grid
        grid_path = Path(self._env._init_grid_path)
        (directory / grid_path.name).write_bytes(grid_path.read_bytes())

        # env general data
        env_data = {
            "n_sub": env.n_sub,
            "n_busbar_per_sub": env.n_busbar_per_sub
        }
        with open(directory / "env.json", "w") as f:
            json.dump(env_data, f, indent=4)

        # one table for each kind of element
        self.write_element_table([env.name_gen, env.gen_type, env.gen_to_subid], ['name', 'type', 'gen_to_subid'], directory, 'gen')
        self.write_element_table([env.name_load, env.load_to_subid], ['name', 'load_to_subid'], directory, 'load')
        self.write_element_table([env.name_shunt, env.shunt_to_subid], ['name', 'shunt_to_subid'], directory, 'shunt')
        self.write_element_table([env.name_line, env.line_or_to_subid, env.line_ex_to_subid], ['name', 'line_or_to_subid', 'line_ex_to_subid'], directory, 'line')

        # one table per element attributs.
        self._tables = [
            ObservationTable(self._env.name_gen, lambda obs: obs.gen_p_before_curtail, directory, 'gen_p_before_curtail', write_chunk_size),
            ObservationTable(self._env.name_gen, lambda obs: obs.gen_p, directory, 'gen_p', write_chunk_size),
            ObservationTable(self._env.name_gen, lambda obs: obs.gen_p_detached, directory, 'gen_p_detached', write_chunk_size),
            ObservationTable(self._env.name_gen, lambda obs: obs.gen_q, directory, 'gen_q', write_chunk_size),
            ObservationTable(self._env.name_gen, lambda obs: obs.gen_bus, directory, 'gen_bus', write_chunk_size),
            ObservationTable(self._env.name_gen, lambda obs: obs.gen_detached, directory, 'gen_detached', write_chunk_size),
            ObservationTable(self._env.name_gen, lambda obs: obs.gen_v, directory, 'gen_v', write_chunk_size),
            ObservationTable(self._env.name_gen, lambda obs: obs.gen_theta, directory, 'gen_theta', write_chunk_size),
            ObservationTable(self._env.name_gen, lambda obs: obs.actual_dispatch, directory, 'gen_actual_dispatch', write_chunk_size),
            ObservationTable(self._env.name_gen, lambda obs: obs.target_dispatch, directory, 'gen_target_dispatch', write_chunk_size),

            ObservationTable(self._env.name_load, lambda obs: obs.load_p, directory, 'load_p', write_chunk_size),
            ObservationTable(self._env.name_load, lambda obs: obs.load_p_detached, directory, 'load_p_detached', write_chunk_size),
            ObservationTable(self._env.name_load, lambda obs: obs.load_q, directory, 'load_q', write_chunk_size),
            ObservationTable(self._env.name_load, lambda obs: obs.load_q_detached, directory, 'load_q_detached', write_chunk_size),
            ObservationTable(self._env.name_load, lambda obs: obs.load_bus, directory, 'load_bus', write_chunk_size),
            ObservationTable(self._env.name_load, lambda obs: obs.load_v, directory, 'load_v', write_chunk_size),
            ObservationTable(self._env.name_load, lambda obs: obs.load_theta, directory, 'load_theta', write_chunk_size),

            ObservationTable(self._env.name_line, lambda obs: obs.line_or_bus, directory, 'line_or_bus', write_chunk_size),
            ObservationTable(self._env.name_line, lambda obs: obs.line_ex_bus, directory, 'line_ex_bus', write_chunk_size),
            ObservationTable(self._env.name_line, lambda obs: obs.line_ex_bus, directory, 'line_status', write_chunk_size),
            ObservationTable(self._env.name_line, lambda obs: obs.p_or, directory, 'line_or_p', write_chunk_size),
            ObservationTable(self._env.name_line, lambda obs: obs.p_ex, directory, 'line_ex_p', write_chunk_size),
            ObservationTable(self._env.name_line, lambda obs: obs.q_or, directory, 'line_or_q', write_chunk_size),
            ObservationTable(self._env.name_line, lambda obs: obs.q_ex, directory, 'line_ex_q', write_chunk_size),
            ObservationTable(self._env.name_line, lambda obs: obs.v_or, directory, 'line_or_v', write_chunk_size),
            ObservationTable(self._env.name_line, lambda obs: obs.v_ex, directory, 'line_ex_v', write_chunk_size),
            ObservationTable(self._env.name_line, lambda obs: obs.theta_or, directory, 'line_or_theta', write_chunk_size),
            ObservationTable(self._env.name_line, lambda obs: obs.theta_ex, directory, 'line_ex_theta', write_chunk_size),
            ObservationTable(self._env.name_line, lambda obs: obs.rho, directory, 'line_rho', write_chunk_size),
            ObservationTable(self._env.name_line, lambda obs: obs.thermal_limit, directory, 'line_thermal_limit', write_chunk_size)
        ]

        self._actions_table = ActionTable(directory, 'actions', write_chunk_size)

    @staticmethod
    def write_element_table(data, column_names, directory: Path, table_name: str):
        element_table = pa.table({col: data[i] for i, col in enumerate(column_names)})
        pa.parquet.write_table(element_table, directory / f"{table_name}.parquet")

    @property
    def env(self):
        return self._env

    def reset(self,
              *,
              seed: Union[int, None] = None,
              options: RESET_OPTIONS_TYPING = None) -> BaseObservation:
        for table in self._tables:
            table.reset()

        self._actions_table.reset()

        obs = self._env.reset(seed=seed, options=options)
        self._append_obs(obs)
        self._actions_table.append(obs.get_time_stamp(), self._env.action_space(), False)
        return obs

    def _append_obs(self, obs: BaseObservation):
        for table in self._tables:
            table.append(obs)

    def step(self, action: BaseAction) -> Tuple[BaseObservation,
                                                float,
                                                bool,
                                                STEP_INFO_TYPING]:
        result = self._env.step(action)
        done = result[2]
        obs = result[0]
        self._append_obs(obs)
        self._actions_table.append(obs.get_time_stamp(), action, done)
        return result

    def render(self, mode="rgb_array"):
        self._env.render(mode=mode)

    def close(self):
        for table in self._tables:
            table.close()

        self._actions_table.close()

        self._env.close()
