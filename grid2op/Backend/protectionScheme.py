import copy
import logging
from typing import Tuple, Union, Any, List, Optional
import numpy as np

from grid2op.dtypes import dt_int
from grid2op.Backend.backend import Backend
from grid2op.Parameters import Parameters
from grid2op.Exceptions import Grid2OpException
from grid2op.Backend.thermalLimits import ThermalLimits

class DefaultProtection:
    """
    Classe avancée pour gérer les protections réseau et les déconnexions.
    """

    def __init__(
        self,
        backend: Backend,
        parameters: Optional[Parameters] = None,
        thermal_limits: Optional[ThermalLimits] = None,
        is_dc: bool = False,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialise l'état du réseau avec des protections personnalisables.
        """
        self.backend = backend
        self._parameters = copy.deepcopy(parameters) if parameters else Parameters()
        self._validate_input(self.backend, self._parameters)

        self.is_dc = is_dc
        self.thermal_limits = thermal_limits

        self._thermal_limit_a = self.thermal_limits.limits if self.thermal_limits else None
        self.backend.thermal_limit_a = self._thermal_limit_a
        
        self._hard_overflow_threshold = self._get_value_from_parameters("HARD_OVERFLOW_THRESHOLD")
        self._soft_overflow_threshold = self._get_value_from_parameters("SOFT_OVERFLOW_THRESHOLD")
        self._nb_timestep_overflow_allowed = self._get_value_from_parameters("NB_TIMESTEP_OVERFLOW_ALLOWED")
        self._no_overflow_disconnection = self._get_value_from_parameters("NO_OVERFLOW_DISCONNECTION")

        self.disconnected_during_cf = np.full(self.thermal_limits.n_line, fill_value=-1, dtype=dt_int)
        self._timestep_overflow = np.zeros(self.thermal_limits.n_line, dtype=dt_int)
        self.conv_ = self._run_power_flow()
        self.infos: List[str] = []

        if logger is None:
            self.logger = logging.getLogger(__name__)
            self.logger.disabled = True
        else:
            self.logger: logging.Logger = logger.getChild("grid2op_BaseEnv")

    def _validate_input(self, backend: Backend, parameters: Optional[Parameters]) -> None:
        if not isinstance(backend, Backend):
            raise Grid2OpException(f"Argument 'backend' doit être de type 'Backend', reçu : {type(backend)}")
        if parameters and not isinstance(parameters, Parameters):
            raise Grid2OpException(f"Argument 'parameters' doit être de type 'Parameters', reçu : {type(parameters)}")

    def _get_value_from_parameters(self, parameter_name: str) -> Any:
        return getattr(self._parameters, parameter_name, None)

    def _run_power_flow(self) -> Optional[Exception]:
        try:
            return self.backend._runpf_with_diverging_exception(self.is_dc)
        except Exception as e:
            if self.logger is not None:
                self.logger.error(
                    f"Erreur flux de puissance : {e}"
                )
            return e

    def _update_overflows(self, lines_flows: np.ndarray) -> np.ndarray:
        if self._thermal_limit_a is None:
            if self.logger is not None:
                self.logger.error(
                    "Thermal limits must be provided for overflow calculations."
                )
            raise ValueError("Thermal limits must be provided for overflow calculations.")

        lines_status = self.backend.get_line_status() # self._thermal_limit_a reste fixe. self._soft_overflow_threshold = 1 
        is_overflowing = (lines_flows >= self._thermal_limit_a * self._soft_overflow_threshold) & lines_status
        self._timestep_overflow[is_overflowing] += 1
        # self._hard_overflow_threshold = 1.5
        exceeds_hard_limit = (lines_flows > self._thermal_limit_a * self._hard_overflow_threshold) & lines_status
        exceeds_allowed_time = self._timestep_overflow > self._nb_timestep_overflow_allowed
        
        lines_to_disconnect = exceeds_hard_limit | (exceeds_allowed_time & lines_status)
        return lines_to_disconnect

    def _disconnect_lines(self, lines_to_disconnect: np.ndarray, timestep: int) -> None:
        for line_idx in np.where(lines_to_disconnect)[0]:
            self.backend._disconnect_line(line_idx)
            self.disconnected_during_cf[line_idx] = timestep
            if self.logger is not None:
                self.logger.warning(f"Ligne {line_idx} déconnectée au pas de temps {timestep}.")         
            

    def next_grid_state(self) -> Tuple[np.ndarray, List[Any], Union[None, Exception]]:
        try:
            timestep = 0
            while True:
                power_flow_result = self._run_power_flow()
                if power_flow_result:
                    return self.disconnected_during_cf, self.infos, power_flow_result

                lines_flows = self.backend.get_line_flow()
                lines_to_disconnect = self._update_overflows(lines_flows)

                if not lines_to_disconnect.any():
                    break

                self._disconnect_lines(lines_to_disconnect, timestep)
                timestep += 1

            return self.disconnected_during_cf, self.infos, None

        except Exception as e:
            if self.logger is not None:
                self.logger.exception("Erreur inattendue dans le calcul de l'état du réseau.")
            return self.disconnected_during_cf, self.infos, e

class NoProtection(DefaultProtection):
    """
    Classe qui désactive les protections de débordement tout en conservant la structure de DefaultProtection.
    """
    def __init__(self, backend: Backend, thermal_limits: ThermalLimits, is_dc: bool = False):
        super().__init__(backend, parameters=None, thermal_limits=thermal_limits, is_dc=is_dc)

    def next_grid_state(self) -> Tuple[np.ndarray, List[Any], None]:
        """
        Ignore les protections et retourne l'état du réseau sans déconnexions.
        """
        return self.disconnected_during_cf, self.infos, self.conv_

class BlablaProtection:
    pass