import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


class NeuronModel:
    """Class containing attributes and methods for the sand-pile model.
    """
    def __init__(self, network, sample_delay=200) -> None:
        # Network should have a node with the label -1 which is the sink node
        # All nodes should be able to reach the sink node
        self.network = network

        assert network.has_node(-1), "Network must contain sink node (label -1)"
        assert nx.ancestors(network, -1) == set(network.nodes).difference({-1,}), "All nodes must be able to reach the sink node"

        self.size = network.number_of_nodes() - 1  # Without the sink node

        self.adj_matrix = nx.adjacency_matrix(self.network, nodelist=[-1] + list(network.nodes)[:-1]).toarray()
        self.potentials = np.zeros(self.adj_matrix.shape[0])
        self.degrees = np.array([np.sum(self.adj_matrix[i]) for i in range(self.adj_matrix.shape[0])])

        self.sample_delay = sample_delay

        # Data storage
        self.avalanche_sizes = []

    def get_node_degree(self, node_i):
        return np.sum(self.adj_matrix[node_i])

    def topple_node(self, node_i):
        new_potentials = np.copy(self.potentials)

        # Decrease potential of toppled node
        new_potentials[node_i] -= self.get_node_degree(node_i)

        # Add one unit of potential to each neighbor
        new_potentials[self.adj_matrix[node_i].astype(bool)] += 1

        self.potentials = new_potentials

    def perform_avalanche(self, start_node):
        unstable = np.array([start_node])
        curr_node = None
        avalanche_size = 0

        while unstable.size > 0:
            # Pick a random unstable node
            random_index = np.random.randint(0, len(unstable))
            curr_node = unstable[random_index]

            # Topple unstable node
            self.topple_node(curr_node)
            avalanche_size += 1

            unstable = np.arange(self.potentials.size)[self.potentials >= self.degrees][1:]

        return avalanche_size

    def step(self, iteration) -> None:
        # Choose random node
        node_i = np.random.randint(1, self.size + 1)

        # Increment potential
        self.potentials[node_i] += 1

        # Check for instabilities
        if self.potentials[node_i] >= self.get_node_degree(node_i):
            avalanche_size = self.perform_avalanche(node_i)

            if iteration > self.sample_delay:
                self.avalanche_sizes.append(avalanche_size)

    def run(self, n_steps):
        assert self.sample_delay < n_steps, "Number of steps must be higher than sample delay"

        for i in range(n_steps):
            self.step(i)

        return np.array(self.avalanche_sizes)


if __name__ == '__main__':
    network = nx.erdos_renyi_graph(100, 0.5, directed=False)
    network.add_node(-1)
    network.add_edge(0, -1)

    model = NeuronModel(network)

    data = model.run(10000)

    plt.hist(data, density=True, log=True)

    plt.title("Avalanche size distribution")
    plt.xlabel("s")
    plt.ylabel("P(s)")

    plt.show()
