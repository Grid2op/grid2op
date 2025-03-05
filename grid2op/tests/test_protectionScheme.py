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
        """Initialisation des mocks et des paramètres de test."""
        self.mock_backend = MagicMock(spec=Backend)
        self.mock_parameters = MagicMock(spec=Parameters)
        self.mock_thermal_limits = MagicMock(spec=ThermalLimits)
        
        # Définition des limites thermiques
        self.mock_thermal_limits.limits = np.array([100.0, 200.0])
        self.mock_thermal_limits.n_line = 2
        
        # Mock des valeurs de paramètre avec des valeurs par défaut
        self.mock_parameters.SOFT_OVERFLOW_THRESHOLD = 1.0
        self.mock_parameters.HARD_OVERFLOW_THRESHOLD = 1.5
        self.mock_parameters.NB_TIMESTEP_OVERFLOW_ALLOWED = 3

        # Comportement du backend
        self.mock_backend.get_line_status.return_value = np.array([True, True])
        self.mock_backend.get_line_flow.return_value = np.array([90.0, 210.0])
        self.mock_backend._runpf_with_diverging_exception.return_value = None

        # Initialisation de la classe testée
        self.default_protection = DefaultProtection(
            backend=self.mock_backend,
            parameters=self.mock_parameters,
            thermal_limits=self.mock_thermal_limits,
            is_dc=False
        )

    def test_initialization(self):
        """Test de l'initialisation de DefaultProtection."""
        self.assertIsInstance(self.default_protection, DefaultProtection)
        self.assertEqual(self.default_protection.is_dc, False)
        self.assertIsNotNone(self.default_protection._parameters)

    def test_validate_input(self):
        """Test de validation des entrées."""
        with self.assertRaises(Grid2OpException):
            DefaultProtection(backend=None, parameters=self.mock_parameters)

    def test_run_power_flow(self):
        """Test de l'exécution du flux de puissance."""
        result = self.default_protection._run_power_flow()
        self.assertIsNone(result)

    def test_update_overflows(self):
        """Test de la mise à jour des surcharges et des lignes à déconnecter."""
        lines_flows = np.array([120.0, 310.0])
        lines_to_disconnect = self.default_protection._update_overflows(lines_flows)
        self.assertTrue(lines_to_disconnect[1])  # Seule la deuxième ligne doit être déconnectée
        self.assertFalse(lines_to_disconnect[0])

    def test_disconnect_lines(self):
        """Test de la déconnexion des lignes."""
        lines_to_disconnect = np.array([False, True])
        self.default_protection._disconnect_lines(lines_to_disconnect, timestep=1)
        self.mock_backend._disconnect_line.assert_called_once_with(1)

    def test_next_grid_state(self):
        """Test de la simulation de l'évolution du réseau."""
        disconnected, infos, error = self.default_protection.next_grid_state()
        self.assertIsInstance(disconnected, np.ndarray)
        self.assertIsInstance(infos, list)
        self.assertIsNone(error)

    def test_no_protection(self):
        """Test de la classe NoProtection."""
        no_protection = NoProtection(self.mock_backend, self.mock_thermal_limits)
        disconnected, infos, conv = no_protection.next_grid_state()
        self.assertIsInstance(disconnected, np.ndarray)
        self.assertIsInstance(infos, list)
        self.assertIsNone(conv)

class TestFunctionalProtection(unittest.TestCase):

    def setUp(self):
        """Initialisation des mocks pour un test fonctionnel."""
        self.mock_backend = MagicMock(spec=Backend)
        self.mock_parameters = MagicMock(spec=Parameters)
        self.mock_thermal_limits = MagicMock(spec=ThermalLimits)

        # Configuration des limites thermiques et des flux de lignes
        self.mock_thermal_limits.limits = np.array([100.0, 200.0])
        self.mock_thermal_limits.n_line = 2
        self.mock_parameters.SOFT_OVERFLOW_THRESHOLD = 1.0
        self.mock_parameters.HARD_OVERFLOW_THRESHOLD = 1.5
        self.mock_parameters.NB_TIMESTEP_OVERFLOW_ALLOWED = 3

        # Comportement du backend pour simuler les flux de lignes
        self.mock_backend.get_line_status.return_value = np.array([True, True])
        self.mock_backend.get_line_flow.return_value = np.array([90.0, 210.0])
        self.mock_backend._runpf_with_diverging_exception.return_value = None

        # Initialisation de la classe de protection avec des paramètres
        self.default_protection = DefaultProtection(
            backend=self.mock_backend,
            parameters=self.mock_parameters,
            thermal_limits=self.mock_thermal_limits,
            is_dc=False
        )

        # Initialisation de NoProtection
        self.no_protection = NoProtection(
            backend=self.mock_backend,
            thermal_limits=self.mock_thermal_limits,
            is_dc=False
        )

    def test_functional_default_protection(self):
        """Test fonctionnel pour DefaultProtection."""

        self.mock_backend.get_line_flow.return_value = np.array([90.0, 210.0])  # Lignes avec un débordement
        disconnected, infos, error = self.default_protection.next_grid_state()

        self.assertTrue(np.any(disconnected == -1))  # Ligne 1 doit être déconnectée
        self.assertIsNone(error)
        self.assertEqual(len(infos), 0)

    def test_functional_no_protection(self):
        """Test fonctionnel pour NoProtection."""
        self.mock_backend.get_line_flow.return_value = np.array([90.0, 180.0])  # Aucune ligne en débordement
        disconnected, infos, error = self.no_protection.next_grid_state()

        self.assertTrue(np.all(disconnected == -1))  # Aucune ligne ne doit être déconnectée
        self.assertIsNone(error)
        self.assertEqual(len(infos), 0)

if __name__ == "__main__":
    unittest.main()
