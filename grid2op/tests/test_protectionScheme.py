import numpy as np

import unittest
from unittest.mock import MagicMock

from grid2op.Backend.backend import Backend
from grid2op.Parameters import Parameters
from grid2op.Exceptions import Grid2OpException

from grid2op.Backend.thermalLimits import ThermalLimits
from grid2op.Backend.protectionScheme import DefaultProtection, NoProtection

class TestProtection(unittest.TestCase):
    
    def setUp(self):
        """Initialization of mocks and test parameters."""
        self.mock_backend = MagicMock(spec=Backend)
        self.mock_parameters = MagicMock(spec=Parameters)
        self.mock_thermal_limits = MagicMock(spec=ThermalLimits)
        
        # Define thermal limits
        self.mock_thermal_limits.limits = np.array([100.0, 200.0])
        self.mock_thermal_limits.n_line = 2
        
        # Default parameters
        self.mock_parameters.SOFT_OVERFLOW_THRESHOLD = 1.0
        self.mock_parameters.HARD_OVERFLOW_THRESHOLD = 1.5
        self.mock_parameters.NB_TIMESTEP_OVERFLOW_ALLOWED = 3

        # Backend behavior
        self.mock_backend.get_line_status.return_value = np.array([True, True])
        self.mock_backend.get_line_flow.return_value = np.array([90.0, 210.0])  # One line is overflowing
        self.mock_backend._runpf_with_diverging_exception.return_value = None

        # Initialize the DefaultProtection class
        self.default_protection = DefaultProtection(
            backend=self.mock_backend,
            parameters=self.mock_parameters,
            thermal_limits=self.mock_thermal_limits,
            is_dc=False
        )
    """
    soft overflow" : with two expected behaviour
    either the line is on overflow for less than "NB_TIMESTEP_OVERFLOW_ALLOWED" (or a similar name not checked in the code) in this case the only consquence is that the overflow counter is increased by 1
    or the line has been on overflow for more than "NB_TIMESTEP_OVERFLOW_ALLOWED" in this case the line is disconnected
    """
    def test_update_overflows_soft(self):
        """Test for soft overflow."""
        lines_flows = np.array([90.0, 210.0])
        lines_to_disconnect = self.default_protection._update_overflows(lines_flows)
        self.assertFalse(lines_to_disconnect[0])  # No disconnection for the first line
        self.assertFalse(lines_to_disconnect[1])  # No disconnection yet, as the overflow is soft

        # il faut se relancer une deuxieme fois et une troiseieme fois pour le deconnecté =3

    def test_update_overflows_hard(self):
        """Test for hard overflow."""
        lines_flows = np.array([120.0, 310.0])
        lines_to_disconnect = self.default_protection._update_overflows(lines_flows)
        self.assertFalse(lines_to_disconnect[0])  # The first line should not be disconnected
        self.assertTrue(lines_to_disconnect[1])  # The second line should be disconnected

    def test_overflow_counter_increase(self):
        """Test that the overflow counter does not exceed 1 per call."""
        self.mock_backend.get_line_flow.return_value = np.array([90.0, 210.0])
        self.default_protection._update_overflows(np.array([90.0, 210.0]))
        self.assertEqual(self.default_protection._timestep_overflow[1], 1)  # The overflow counter for line 1 should be 1

        # Next call with different flow
        self.mock_backend.get_line_flow.return_value = np.array([90.0, 220.0])
        self.default_protection._update_overflows(np.array([90.0, 220.0]))
        self.assertEqual(self.default_protection._timestep_overflow[1], 2)  # The overflow counter for line 1 should be 2

    def test_initialization(self):
        """Test the initialization of DefaultProtection."""
        self.assertIsInstance(self.default_protection, DefaultProtection)
        self.assertEqual(self.default_protection.is_dc, False)
        self.assertIsNotNone(self.default_protection._parameters)

    def test_validate_input(self):
        """Test input validation."""
        with self.assertRaises(Grid2OpException):
            DefaultProtection(backend=None, parameters=self.mock_parameters)

    def test_run_power_flow(self):
        """Test running the power flow."""
        result = self.default_protection._run_power_flow()
        self.assertIsNone(result)

    def test_update_overflows(self):
        """Test updating overflows and lines to disconnect."""
        lines_flows = np.array([120.0, 310.0])
        lines_to_disconnect = self.default_protection._update_overflows(lines_flows)
        self.assertTrue(lines_to_disconnect[1])  # Only the second line should be disconnected
        self.assertFalse(lines_to_disconnect[0])

    def test_disconnect_lines(self):
        """Test disconnecting lines."""
        lines_to_disconnect = np.array([False, True])
        self.default_protection._disconnect_lines(lines_to_disconnect, timestep=1)
        self.mock_backend._disconnect_line.assert_called_once_with(1)

    def test_next_grid_state(self):
        """Test simulating the network's evolution."""
        disconnected, infos, error = self.default_protection.next_grid_state()
        self.assertIsInstance(disconnected, np.ndarray)
        self.assertIsInstance(infos, list)
        self.assertIsNone(error)

    def test_no_protection(self):
        """Test the NoProtection class."""
        no_protection = NoProtection(self.mock_backend, self.mock_thermal_limits)
        disconnected, infos, conv = no_protection.next_grid_state()
        self.assertIsInstance(disconnected, np.ndarray)
        self.assertIsInstance(infos, list)
        self.assertIsNone(conv)

class TestFunctionalProtection(unittest.TestCase):

    def setUp(self):
        """Initialization of mocks for a functional test."""
        self.mock_backend = MagicMock(spec=Backend)
        self.mock_parameters = MagicMock(spec=Parameters)
        self.mock_thermal_limits = MagicMock(spec=ThermalLimits)

        # Set thermal limits and line flows
        self.mock_thermal_limits.limits = np.array([100.0, 200.0])
        self.mock_thermal_limits.n_line = 2
        self.mock_parameters.SOFT_OVERFLOW_THRESHOLD = 1.0
        self.mock_parameters.HARD_OVERFLOW_THRESHOLD = 1.5
        self.mock_parameters.NB_TIMESTEP_OVERFLOW_ALLOWED = 3

        # Backend behavior to simulate line flows
        self.mock_backend.get_line_status.return_value = np.array([True, True])
        self.mock_backend.get_line_flow.return_value = np.array([90.0, 210.0])
        self.mock_backend._runpf_with_diverging_exception.return_value = None

        # Initialize protection class with parameters
        self.default_protection = DefaultProtection(
            backend=self.mock_backend,
            parameters=self.mock_parameters,
            thermal_limits=self.mock_thermal_limits,
            is_dc=False
        )

        # Initialize NoProtection
        self.no_protection = NoProtection(
            backend=self.mock_backend,
            thermal_limits=self.mock_thermal_limits,
            is_dc=False
        )

    def test_functional_default_protection(self):
        """Functional test for DefaultProtection."""
        self.mock_backend.get_line_flow.return_value = np.array([90.0, 210.0])  # Lines with overflow
        disconnected, infos, error = self.default_protection.next_grid_state()

        self.assertTrue(np.any(disconnected == -1))  # Line 1 should be disconnected
        self.assertIsNone(error)
        self.assertEqual(len(infos), 0)

    def test_functional_no_protection(self):
        """Functional test for NoProtection."""
        self.mock_backend.get_line_flow.return_value = np.array([90.0, 180.0])  # No lines overflowing
        disconnected, infos, error = self.no_protection.next_grid_state()

        self.assertTrue(np.all(disconnected == -1))  # No line should be disconnected
        self.assertIsNone(error)
        self.assertEqual(len(infos), 0)

if __name__ == "__main__":
    unittest.main()