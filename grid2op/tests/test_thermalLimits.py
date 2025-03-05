import unittest
import numpy as np
import logging
from grid2op.Backend.protectionScheme import ThermalLimits
from grid2op.Exceptions import Grid2OpException

class TestThermalLimits(unittest.TestCase):
    
    def setUp(self):
        """Initialisation avant chaque test."""
        self.logger = logging.getLogger("test_logger")
        self.logger.disabled = True
        
        self.n_lines = 3
        self.line_names = ["Line1", "Line2", "Line3"]
        self.thermal_limits = np.array([100.0, 200.0, 300.0])
        
        self.thermal_limit_instance = ThermalLimits(
            _thermal_limit_a=self.thermal_limits,
            line_names=self.line_names,
            n_line=self.n_lines,
            logger=self.logger
        )

    def test_initialization(self):
        """Test de l'initialisation de ThermalLimits."""
        self.assertEqual(self.thermal_limit_instance.n_line, self.n_lines)
        self.assertEqual(self.thermal_limit_instance.name_line, self.line_names)
        np.testing.assert_array_equal(self.thermal_limit_instance.limits, self.thermal_limits)

    def test_set_n_line(self):
        """Test du setter de n_line."""
        self.thermal_limit_instance.n_line = 5
        self.assertEqual(self.thermal_limit_instance.n_line, 5)

        with self.assertRaises(ValueError):
            self.thermal_limit_instance.n_line = -1  # Doit lever une erreur

    def test_set_name_line(self):
        """Test du setter de name_line."""
        new_names = ["L4", "L5", "L6"]
        self.thermal_limit_instance.name_line = new_names
        self.assertEqual(self.thermal_limit_instance.name_line, new_names)

        with self.assertRaises(ValueError):
            self.thermal_limit_instance.name_line = ["L4", 123, "L6"]  # Doit lever une erreur

    def test_set_limits(self):
        """Test du setter de limits avec np.array et dict."""
        new_limits = np.array([400.0, 500.0, 600.0])
        self.thermal_limit_instance.limits = new_limits
        np.testing.assert_array_equal(self.thermal_limit_instance.limits, new_limits)

        limit_dict = {"Line1": 110.0, "Line2": 220.0, "Line3": 330.0}
        self.thermal_limit_instance.limits = limit_dict
        np.testing.assert_array_equal(
            self.thermal_limit_instance.limits, np.array([110.0, 220.0, 330.0])
        )

        with self.assertRaises(Grid2OpException):
            self.thermal_limit_instance.limits = {"InvalidLine": 100.0}  # Ligne inexistante

    def test_copy(self):
        """Test de la méthode copy."""
        copied_instance = self.thermal_limit_instance.copy()
        self.assertIsNot(copied_instance, self.thermal_limit_instance)
        np.testing.assert_array_equal(copied_instance.limits, self.thermal_limit_instance.limits)

if __name__ == "__main__":
    unittest.main()
