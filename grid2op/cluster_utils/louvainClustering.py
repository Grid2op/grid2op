from grid2op.Environment import Environment
import numpy as np
from sknetwork.clustering import Louvain
from scipy.sparse import csr_matrix

class LouvainClustering:
    """
    Outputs clustered substation based on the Louvain graph clustering method.
    """
    
    # Create connectivity matrix
    @staticmethod
    def create_connectivity_matrix(env:Environment):
        """
        Creates a connectivity matrix for the given grid environment.

        The connectivity matrix is a 2D NumPy array where the element at position (i, j) is 1 if there is a direct 
        connection between substation i and substation j, and 0 otherwise. The diagonal elements are set to 1 to indicate 
        self-connections.

        Args:
            env (grid2op.Environment): The grid environment for which the connectivity matrix is to be created.

        Returns:
            connectivity_matrix: A 2D Numpy array of dimension (env.n_sub, env.n_sub) representing the substation connectivity of the grid environment.
        """
        connectivity_matrix = np.zeros((env.n_sub, env.n_sub))
        for line_id in range(env.n_line):
            orig_sub = env.line_or_to_subid[line_id]
            extrem_sub = env.line_ex_to_subid[line_id]
            connectivity_matrix[orig_sub, extrem_sub] = 1
            connectivity_matrix[extrem_sub, orig_sub] = 1
        return connectivity_matrix + np.eye(env.n_sub)

    
       
    # Cluster substations
    @staticmethod
    def cluster_substations(env:Environment):
        """
        Clusters substations in a power grid environment using the Louvain community detection algorithm.

        This function generates a connectivity matrix representing the connections between substations in the given environment, 
        and applies the Louvain algorithm to cluster the substations into communities. The resulting clusters are formatted into 
        a dictionary where each key corresponds to an agent and the value is a list of substations assigned to that agent.

        Args:
            env (grid2op.Environment): The grid environment for which the connectivity matrix is to be created.
            
        Returns:
                (MADict):
                    - keys : agents' names 
                    - values : list of substations' id under the control of the agent.
        """

        # Generate the connectivity matrix
        matrix = LouvainClustering.create_connectivity_matrix(env)

        # Perform clustering using Louvain algorithm
        louvain = Louvain()
        adjacency = csr_matrix(matrix)
        labels = louvain.fit_predict(adjacency)

        # Group substations into clusters
        clusters = {}
        for node, label in enumerate(labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(node)

        # Format the clusters
        formatted_clusters = {f'agent_{i}': nodes for i, nodes in enumerate(clusters.values())}
        
        return formatted_clusters