from grid2op.Environment import Environment
import numpy as np
from scipy.sparse import csr_matrix

# Try/except for scikit-network imports
try:
    from sknetwork.clustering import Louvain
except ImportError:
    Louvain = None
    import warnings
    warnings.warn(" scikit-network is not installed. Louvain clustering will not be available.")

class LouvainClustering:
    """
    Clusters substations in a power grid environment using the Louvain community detection algorithm.

    This class provides functionality to analyze the electrical connectivity between substations in a 
    Grid2Op environment and cluster them into communities based on their structural proximity and 
    interaction within the grid network. 

    The Louvain algorithm is used for this purpose due to its efficiency, scalability, and ability to 
    detect high modularity communities — areas where substations are densely connected internally and 
    sparsely connected externally. These characteristics make the algorithm particularly well-suited for 
    substation clustering, as it helps form cohesive subgrids, enabling more efficient and optimized 
    multi-agent operation strategies.

    Key Features:
    - Builds a substation-level connectivity matrix based on line connections in the environment.
    - Applies the Louvain algorithm to detect community structures (clusters of substations).
    - Returns a dictionary that maps agents to groups of clustered substations.

    The Louvain algorithm operates in two main iterative phases:
    1. **Partitioning**: Nodes are greedily assigned to communities to maximize modularity.
    2. **Aggregation**: Discovered communities are treated as super-nodes and the process is repeated 
       until no further modularity gain is possible.

    Note:
        scikit-network must be installed for this clustering method to function.
    
    References:
        More information on the Louvain Algorithm is available at https://web.stanford.edu/class/cs246/slides/11-graphs1.pdf
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
        # Check if Louvain is available
        if Louvain is None:
            raise ImportError("scikit-network is required for Louvain clustering but is not installed.")

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
    
    @staticmethod
    def is_louvain_available():
        """
        Checks if the Louvain algorithm is available.
        
        Returns:
            bool: True if Louvain is available, False otherwise.
        """
        return Louvain is not None