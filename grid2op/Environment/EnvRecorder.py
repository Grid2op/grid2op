# Copyright (c) 2025, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.
from abc import ABC
from datetime import datetime
from pathlib import Path
from typing import Tuple, Union, Callable, List

from grid2op.Action import BaseAction, baseAction
from grid2op.Environment.EnvInterface import EnvInterface
from grid2op.Observation import BaseObservation
from grid2op.typing_variables import STEP_INFO_TYPING, RESET_OPTIONS_TYPING
import pyarrow as pa
import pyarrow.parquet


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
        super().__init__(['action'], directory, table_name, write_chunk_size)

    def append(self, time: datetime, act: BaseAction):
        self._buffer[0].append(int(time.timestamp()))
        self._buffer[1].append(str(act.as_dict()))
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

        # one table for each kind of element
        self.write_element_table([env.name_gen, env.gen_type], ['name', 'type'], directory, 'gen')
        self.write_element_table([env.name_load], ['name'], directory, 'load')

        # one table per element attributs.
        self._tables = [
            ObservationTable(self._env.name_gen, lambda obs: obs.gen_p_before_curtail, directory, 'gen_p_before_curtail', write_chunk_size),
            ObservationTable(self._env.name_gen, lambda obs: obs.gen_p, directory, 'gen_p', write_chunk_size),
            ObservationTable(self._env.name_gen, lambda obs: obs.gen_v, directory, 'gen_v', write_chunk_size),
            ObservationTable(self._env.name_load, lambda obs: obs.load_p, directory, 'load_p', write_chunk_size),
            ObservationTable(self._env.name_load, lambda obs: obs.load_q, directory, 'load_q', write_chunk_size)
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

        return self._env.reset(seed=seed, options=options)

    def step(self, action: BaseAction) -> Tuple[BaseObservation,
                                                float,
                                                bool,
                                                STEP_INFO_TYPING]:
        result = self._env.step(action)
        obs = result[0]

        for table in self._tables:
            table.append(obs)

        self._actions_table.append(obs.get_time_stamp(), action)

        return result

    def render(self, mode="rgb_array"):
        self._env.render(mode=mode)

    def close(self):
        for table in self._tables:
            table.close()

        self._actions_table.close()

        self._env.close()
