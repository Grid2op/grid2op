import unittest
import numpy as np
import grid2op
from lightsim2grid.lightSimBackend import LightSimBackend

# Try to import LouvainClustering
try:
    from grid2op.multi_agent import LouvainClustering
    # Check if Louvain is available within the class
    louvain_available = LouvainClustering.is_louvain_available()
except ImportError:
    louvain_available = False

class TestLouvainClustering(unittest.TestCase):

    def test_create_connectivity_matrix(self):
        """
        Test the creation of the connectivity matrix
        """
        
        if not louvain_available:
            self.skipTest("Louvain algorithm is not available. Skipping test.")

        
        env_name = "l2rpn_case14_sandbox"
        env = grid2op.make(env_name, backend=LightSimBackend(), test=True)
        
        # Expected connectivity matrix 
        expected_matrix = np.array([
            [1., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 1., 1., 1., 1., 0., 1., 0., 1., 0., 0., 0., 0., 0.],
            [1., 1., 0., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 1., 1., 1., 0.],
            [0., 0., 0., 1., 0., 0., 1., 1., 1., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 1., 0., 0., 1., 0., 1., 1., 0., 0., 0., 1.],
            [0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 0., 0., 0.],
            [0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 1., 0., 0., 0.],
            [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 1., 0.],
            [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 1., 1.],
            [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 1.]
        ])
        
        # Generate the connectivity matrix
        actual_matrix = LouvainClustering.create_connectivity_matrix(env)
    

        # Validate the generated matrix
        np.testing.assert_array_almost_equal(actual_matrix, expected_matrix, err_msg="Connectivity matrix does not match the expected matrix.")

        print("Test passed for create_connectivity_matrix.")
        
        
    def test_cluster_substations(self):
        """
        Test the clustering of substations using the Louvain algorithm
        """
        
        if not louvain_available:
            self.skipTest("Louvain algorithm is not available. Skipping test.")
        
        env_name = "l2rpn_case14_sandbox"
        env = grid2op.make(env_name, backend=LightSimBackend(), test=True)
        
        # Expected clustering result
        expected_clusters = {
            'agent_0': [0, 1, 2, 3, 4],
            'agent_1': [5, 11, 12],
            'agent_2': [6, 7, 8, 13],
            'agent_3': [9, 10]
        }
        
        # Generate the clustering
        actual_clusters = LouvainClustering.cluster_substations(env)
        
        # Validate the generated clustering
        self.assertEqual(
            actual_clusters,
            expected_clusters,
            f"Clustered substations do not match the expected result. Got {actual_clusters}"
        )

        print("Test passed for cluster_substations.")

if __name__ == '__main__':
    unittest.main()
    
    
