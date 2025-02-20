from typing import Union, List, Optional, Dict
try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

import logging
import numpy as np
import copy

import grid2op
from grid2op.dtypes import dt_float
from grid2op.Exceptions import Grid2OpException

class ThermalLimits:
    """
    Class for managing the thermal limits of power grid lines.
    """

    def __init__(
            self,
            _thermal_limit_a: Optional[np.ndarray] = None,
            line_names: Optional[List[str]] = None,
            n_line: Optional[int] = None,
            logger: Optional[logging.Logger] = None,
        ):
        """
        Initializes the thermal limits manager.

        :param thermal_limits: Optional[np.ndarray]
            Array of thermal limits for each power line. Must have the same length as the number of lines.
        :param line_names: Optional[List[str]]
            List of power line names.
        :param n_line: Optional[int]
            Number of lines (can be passed explicitly or inferred from `thermal_limits` or `line_names`).

        :raises ValueError:
            If neither `thermal_limits` nor `n_line` and `line_names` are provided.
        """
        if _thermal_limit_a is None and (n_line is None and line_names is None):
            raise ValueError("Must provide thermal_limits or both n_line and line_names.")

        self._thermal_limit_a = _thermal_limit_a
        self._n_line = n_line
        self._name_line = line_names

        if logger is None:
            self.logger = logging.getLogger(__name__)
            self.logger.disabled = True
        else:
            self.logger: logging.Logger = logger.getChild("grid2op_BaseEnv")
        
    @property
    def n_line(self) -> int:
        return self._n_line

    @n_line.setter
    def n_line(self, new_n_line: int) -> None:
        if new_n_line <= 0:
            raise ValueError("Number of lines must be a positive integer.")
        self._n_line = new_n_line
        if self.logger is not None:
            self.logger.info(f"Number of lines updated to {self._n_line}.")

    @property
    def name_line(self) -> Union[List[str], np.ndarray]:
        return self._name_line

    @name_line.setter
    def name_line(self, new_name_line: Union[List[str], np.ndarray]) -> None:
        if isinstance(new_name_line, np.ndarray):
            if not np.all([isinstance(name, str) for name in new_name_line]):
                raise ValueError("All elements in name_line must be strings.")
        elif isinstance(new_name_line, list):
            if not all(isinstance(name, str) for name in new_name_line):
                raise ValueError("All elements in name_line must be strings.")
        else:
            raise ValueError("Line names must be a list or numpy array of non-empty strings.")
        
        if self._n_line is not None and len(new_name_line) != self._n_line:
            raise ValueError("Length of name list must match the number of lines.")

        self._name_line = new_name_line
        if self.logger is not None:
            self.logger.info("Power line names updated")

    @property
    def limits(self) -> np.ndarray:
        """
        Gets the current thermal limits of the power lines.

        :return: np.ndarray
            The array containing thermal limits for each power line.
        """
        return self._thermal_limit_a

    @limits.setter
    def limits(self, new_limits: Union[np.ndarray, Dict[str, float]]):
        """
        Sets new thermal limits.

        :param new_limits: Union[np.ndarray, Dict[str, float]]
            Either a numpy array or a dictionary mapping line names to new thermal limits.

        :raises ValueError:
            If the new limits array size does not match the number of lines.
        :raises Grid2OpException:
            If invalid power line names are provided in the dictionary.
            If the new thermal limit values are invalid (non-positive or non-convertible).
        :raises TypeError:
            If the input type is not an array or dictionary.
        """
        if isinstance(new_limits, np.ndarray):
            if new_limits.shape[0] == self.n_line:
                self._thermal_limit_a = 1.0 * new_limits.astype(dt_float)
        elif isinstance(new_limits, dict):
            for el in new_limits.keys():
                if not el in self.name_line:
                    raise Grid2OpException(
                        'You asked to modify the thermal limit of powerline named "{}" that is not '
                        "on the grid. Names of powerlines are {}".format(
                            el, self.name_line
                        )
                    )
            for i, el in self.name_line:
                if el in new_limits:
                    try:
                        tmp = dt_float(new_limits[el])
                    except Exception as exc_:
                        raise Grid2OpException(
                            'Impossible to convert data ({}) for powerline named "{}" into float '
                            "values".format(new_limits[el], el)
                        ) from exc_
                    if tmp <= 0:
                        raise Grid2OpException(
                            'New thermal limit for powerlines "{}" is not positive ({})'
                            "".format(el, tmp)
                        )
                    self._thermal_limit_a[i] = tmp

    def env_limits(self, thermal_limit):
        """
        """
        if isinstance(thermal_limit, dict):
            tmp = np.full(self.n_line, fill_value=np.NaN, dtype=dt_float)
            for key, val in thermal_limit.items():
                if key not in self.name_line:
                    raise Grid2OpException(
                        f"When setting a thermal limit with a dictionary, the keys should be line "
                        f"names. We found: {key} which is not a line name. The names of the "
                        f"powerlines are {self.name_line}"
                    )
                ind_line = (self.name_line == key).nonzero()[0][0]
                if np.isfinite(tmp[ind_line]):
                    raise Grid2OpException(
                        f"Humm, there is a really strange bug, some lines are set twice."
                    )
                try:
                    val_fl = float(val)
                except Exception as exc_:
                    raise Grid2OpException(
                        f"When setting thermal limit with a dictionary, the keys should be "
                        f"the values of the thermal limit (in amps) you provided something that "
                        f'cannot be converted to a float. Error was "{exc_}".'
                    )
                tmp[ind_line] = val_fl

        elif isinstance(thermal_limit, (np.ndarray, list)):
            try:
                tmp = np.array(thermal_limit).flatten().astype(dt_float)
            except Exception as exc_:
                raise Grid2OpException(
                    f"Impossible to convert the vector as input into a 1d numpy float array. "
                    f"Error was: \n {exc_}"
                )
            if tmp.shape[0] != self.n_line:
                raise Grid2OpException(
                    "Attempt to set thermal limit on {} powerlines while there are {}"
                    "on the grid".format(tmp.shape[0], self.n_line)
                )
            if (~np.isfinite(tmp)).any():
                raise Grid2OpException(
                    "Impossible to use non finite value for thermal limits."
                )
        else:
            raise Grid2OpException(
                f"You can only set the thermal limits of the environment with a dictionary (in that "
                f"case the keys are the line names, and the values the thermal limits) or with "
                f"a numpy array that has as many components of the number of powerlines on "
                f'the grid. You provided something with type "{type(thermal_limit)}" which '
                f"is not supported."
            )

        self._thermal_limit_a = tmp
        if self.logger is not None:
            self.logger.info("Env thermal limits successfully set.")

    def update_limits_from_vector(self, thermal_limit_a: np.ndarray) -> None:
        """
        Updates the thermal limits using a numpy array.

        :param thermal_limit_a: np.ndarray
            The new array of thermal limits (in Amperes).
        """
        thermal_limit_a = np.array(thermal_limit_a).astype(dt_float)
        self._thermal_limit_a = thermal_limit_a
        if self.logger is not None:
            self.logger.info("Thermal limits updated from vector.")

    def update_limits(self, env : "grid2op.Environment.BaseEnv") -> None:
        pass

    def copy(self) -> Self:
        """
        Creates a deep copy of the current ThermalLimits instance.

        :return: ThermalLimits
            A new instance with the same attributes.
        """
        return copy.deepcopy(self)