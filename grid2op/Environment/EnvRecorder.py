from datetime import datetime
from pathlib import Path
from typing import Tuple, Union, Callable, Dict, List

from grid2op.Action import BaseAction
from grid2op.Environment.EnvInterface import EnvInterface
from grid2op.Observation import BaseObservation
from grid2op.typing_variables import STEP_INFO_TYPING, RESET_OPTIONS_TYPING
import pyarrow as pa
import pyarrow.parquet

ObservationVectorGetter = Callable[[BaseObservation], List[float]]

class ObservationTable:
    def __init__(self, columns: List[str], getter: ObservationVectorGetter, directory: Path, table_name: str, write_chunk_size: int):
        self._columns = columns
        self._getter = getter
        self._directory = directory
        self._table_name = table_name
        self._write_chunk_size = write_chunk_size
        self._buffer = [[] for _ in range(len(columns) + 1)]
        self._writer = None

    def append(self, obs: BaseObservation):
        time = obs.get_time_stamp()
        self._buffer[0].append(int(time.timestamp()))
        vec = self._getter(obs)
        for i in range(len(self._columns)):
            self._buffer[i + 1].append(vec[i])

        if len(self._buffer[0]) >= self._write_chunk_size:
            table = pa.table(self._buffer, ['time'] + list(self._columns))
            if self._writer is None:
                parquet_file = self._directory / f"{self._table_name}.parquet"
                self._writer = pa.parquet.ParquetWriter(parquet_file, schema=table.schema)
            self._writer.write_table(table)
            self._buffer = [[] for _ in range(len(self._columns) + 1)] # reset buffer

    def close(self):
        if self._writer is not None:
            self._writer.close()
            self._writer = None

class EnvRecorder(EnvInterface):

    def __init__(self, env, directory: Path, write_chunk_size: int = 1000):
        super().__init__()
        self._env = env

        self.write_element_table([env.name_gen, env.gen_type], ['name', 'type'], directory, 'gen')
        self.write_element_table([env.name_load], ['name'], directory, 'load')

        self._gen_p_before_curtail_table = ObservationTable(self._env.name_gen, lambda obs: obs.gen_p_before_curtail, directory, 'gen_p_before_curtail', write_chunk_size)
        self._gen_p_table = ObservationTable(self._env.name_gen, lambda obs: obs.gen_p, directory, 'gen_p', write_chunk_size)
        self._gen_v_table = ObservationTable(self._env.name_gen, lambda obs: obs.gen_v, directory, 'gen_v', write_chunk_size)
        self._load_p_table = ObservationTable(self._env.name_load, lambda obs: obs.load_p, directory, 'load_p', write_chunk_size)
        self._load_q_table = ObservationTable(self._env.name_load, lambda obs: obs.load_q, directory, 'load_q', write_chunk_size)

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
        return self._env.reset(seed=seed, options=options)

    def step(self, action: BaseAction) -> Tuple[BaseObservation,
                                                float,
                                                bool,
                                                STEP_INFO_TYPING]:
        result = self._env.step(action)
        obs = result[0]
#        print(result[3]['time_series_id'])
        self._gen_p_before_curtail_table.append(obs)
        self._gen_p_table.append(obs)
        self._gen_v_table.append(obs)
        self._load_p_table.append(obs)
        self._load_q_table.append(obs)
        return result

    def render(self, mode="rgb_array"):
        self._env.render(mode=mode)

    def close(self):
        self._gen_p_before_curtail_table.close()
        self._gen_p_table.close()
        self._gen_v_table.close()
        self._load_p_table.close()
        self._load_q_table.close()
        self._env.close()
