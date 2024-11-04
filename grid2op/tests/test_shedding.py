import warnings
import unittest
import grid2op
from grid2op.Parameters import Parameters

class TestShedding(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()
        p = Parameters()
        p.ALLOW_SHEDDING = True
        p.MAX_SUB_CHANGED = 5
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make("rte_case5_example", param=p, test=True)

    def tearDown(self) -> None:
        self.env.close()

    def test_shedding_parameter_is_true(self):
        assert self.env.parameters.ALLOW_SHEDDING is True

    def test_shed_single_load(self):
        # Check that a single load can be shed
        pass

    def test_shed_single_generator(self):
        # Check that a single generator can be shed
        pass

    def test_shed_multiple_loads(self):
        # Check that multiple loads can be shed at the same time
        pass

    def test_shed_multiple_generators(self):
        # Check that multiple generators can be shed at the same time
        pass

    def test_shed_load_and_generator(self):
        # Check that load and generator can be shed at the same time
        pass

    def test_shed_energy_storage(self):
        # Check that energy storage device can be shed successfully
        pass

    def test_shedding_appears_in_observation(self):
        # Check that observation contains correct information about shedding
        pass

    def test_shedding_persistance(self):
        # Check that components remains disconnected if shed
        pass

if __name__ == "__main__":
    unittest.main()
