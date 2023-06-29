import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


class NeuronModel:
    """Class containing attributes and methods for the sand-pile model.
    """
    def __init__(self, network, sample_delay=0, start_filled=False) -> None:
        # Network should have a node with the label -1 which is the sink node
        # All nodes should be able to reach the sink node
        self.network = network

        assert network.has_node(-1), "Network must contain sink node (label -1)"
        assert nx.ancestors(network, -1) == set(network.nodes).difference({-1,}), "All nodes must be able to reach the sink node"

        self.size = network.number_of_nodes() - 1  # Without the sink node

        self.adj_matrix = nx.adjacency_matrix(self.network, nodelist=[-1] + list(network.nodes)[:-1]).toarray()

        if start_filled:
            self.potentials = [x for (_, x) in self.network.degree()]
            self.potentials = [self.potentials[-1]] + self.potentials[:-1]
            self.potentials = self.potentials - np.ones_like(self.potentials)
        else:
            self.potentials = np.zeros(self.adj_matrix.shape[0], dtype=int)
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
            curr_node = np.random.choice(unstable)

            # Topple unstable node
            self.topple_node(curr_node)
            avalanche_size += 1

            unstable = np.arange(self.potentials.size)[self.potentials >= self.degrees]
            unstable = unstable[unstable != 0]

        return avalanche_size

    def step(self, iteration) -> None:
        # Choose random node (ignore node 0 (sink node))
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


def create_2d_grid_graph(rows, columns):
    """Returns network as 2D grid graph and nodes on the periphery are sinks"""
    grid_graph = nx.grid_2d_graph(rows, columns, create_using=nx.MultiGraph)
    converted_graph = nx.convert_node_labels_to_integers(grid_graph)
    converted_graph.add_node(-1)  # Add sink node

    for node in converted_graph.nodes():
        for _ in range(4 - converted_graph.degree(node)):
            converted_graph.add_edge(node, -1)  # Add edge to sink node

    color_map =  ['#1f78b4' for _ in range(25)] + ['black']
    pos = {i: (y,-x) for i, (x, y) in enumerate(grid_graph.nodes())}
    pos[-1] = (-1, -1)


    nx.draw(converted_graph, pos=pos, node_color=color_map)
    plt.show()

    return converted_graph


def create_random_graph(size, p, random_sink_node_edges=False):
    network = nx.erdos_renyi_graph(size, p, directed=False)

    components = [c for c in nx.connected_components(network)]
    network.add_node(-1)
    for comp in components:
        network.add_edge(list(comp)[0], -1)

    if random_sink_node_edges:
        for node in network.nodes():
            if np.random.rand() < p:
                network.add_edge(node, -1)

    return network


if __name__ == '__main__':
    network = create_random_graph(100, 0.5, random_sink_node_edges=True)
    model = NeuronModel(network, sample_delay=0, start_filled=True)

    data = model.run(100000)

    avalanche_sizes_grid, frequencies_grid = np.unique(data, return_counts=True)

    # Plot the data points on a log-log scale
    plt.figure()
    plt.scatter(avalanche_sizes_grid, frequencies_grid )

    plt.title("Avalanche size distribution")
    plt.xlabel("s")
    plt.ylabel("P(s)")

    # Set log-log scale
    plt.xscale('log')
    plt.yscale('log')


    plt.show()
