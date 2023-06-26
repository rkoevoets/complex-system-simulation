import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


class NeuronModel:
    """Class containing attributes and methods for the sand-pile model.
    """
    def __init__(self, size, sample_delay=200) -> None:
        self.network = nx.erdos_renyi_graph(size, 0.5, directed=False)
        self.size = size  # Without the sink node
        self.sample_delay = sample_delay

        # Add sink nodes
        self.network.add_node(-1)
        self.network.add_edge(-1, 0)
        self.network.add_edge(-1, 5)

        # Initialise potentials
        init_potentials = {i: 0.0 for i in range(-1, size)}
        nx.set_node_attributes(self.network, init_potentials, name="potential")

        # Data storage
        self.avalanche_sizes = []

    def get_neuron_degree(self, node_i):
        return nx.degree(self.network)[node_i]

    def get_neuron_potential(self, node_i):
        all_potentials = nx.get_node_attributes(self.network, "potential")

        return all_potentials[node_i]

    def add_neuron_potential(self, node_i, potential):
        curr_potential = self.get_neuron_potential(node_i)

        nx.set_node_attributes(self.network, {node_i: curr_potential + potential}, "potential")

    def set_neuron_potential(self, node_i, potential):
        nx.set_node_attributes(self.network, {node_i: potential}, "potential")

    def get_neighbors(self, node_i):
        return list(self.network.neighbors(node_i))

    def pick_random_non_sink(self):
        node = list(self.network.nodes)[np.random.randint(0, self.size)]

        return node

    def topple_node(self, node_i):
        neighbors = self.get_neighbors(node_i)

        potentials = nx.get_node_attributes(self.network, "potential")
        new_potentials = {neigh: potentials[neigh] + 1 for neigh in neighbors}
        new_potentials[node_i] = potentials[node_i] - len(list(self.get_neighbors(node_i)))

        nx.set_node_attributes(self.network, new_potentials, "potential")

    def perform_avalanche(self, start_node):
        unstable = [start_node]
        curr_node = None
        avalanche_size = 0

        while unstable:
            random_index = np.random.randint(0, len(unstable))
            curr_node = unstable.pop(random_index)

            # Topple unstable node
            self.topple_node(curr_node)
            avalanche_size += 1

            # Check if neighbors are now unstable
            # If yes, add to queue
            for neigh in self.get_neighbors(curr_node):
                potential = self.get_neuron_potential(neigh)
                degree = self.get_neuron_degree(neigh)

                # Skip nodes with degree zero
                if degree == 0:
                    continue

                # Skip sink node
                if neigh == -1:
                    continue

                # Skip if it is already unstable
                if neigh in unstable:
                    continue

                # Node is unstable
                if potential >= degree:
                    unstable.append(neigh)

        return avalanche_size

    def step(self, iteration) -> None:
        # Choose random node
        node_i = self.pick_random_non_sink()

        # Increment grains
        self.add_neuron_potential(node_i, 1.0)

        # Check for instabilities
        if self.get_neuron_potential(node_i) >= self.get_neuron_degree(node_i):
            avalanche_size = self.perform_avalanche(node_i)

            if iteration > self.sample_delay:
                self.avalanche_sizes.append(avalanche_size)


    def run(self, n_steps):
        assert self.sample_delay < n_steps, "Number of steps must be higher than sample delay"

        for i in range(n_steps):
            self.step(i)

        return self.avalanche_sizes


if __name__ == '__main__':
    model = NeuronModel(20)

    data = np.array(model.run(1000))

    print(data)

    # plt.hist(data, density=True, log=True)

    # plt.title("Avalanche size distribution")
    # plt.xlabel("s")
    # plt.ylabel("P(s)")

    # plt.show()
