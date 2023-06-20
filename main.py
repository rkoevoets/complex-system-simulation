import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


class NeuronModel:
    """Class containing attributes and methods for the sand-pile model.
    """
    def __init__(self, size) -> None:
        self.network = nx.erdos_renyi_graph(size, 0.5, directed=True)
        self.t_max = 10

        # Initialise potentials
        init_potentials = {i: -1.0 for i in range(size)}
        nx.set_node_attributes(self.network, init_potentials, name="potential")

    def get_neuron_potential(self, node_i):
        all_potentials = nx.get_node_attributes(self.network, "potential")

        return all_potentials[node_i]

    def add_neuron_potential(self, node_i, potential):
        curr_potential = self.get_neuron_potential(node_i)

        nx.set_node_attributes(self.network, {node_i: curr_potential + potential})

    def get_neighbors(self, node_i):
        return self.network.neighbors(node_i)

    def step(self) -> None:
        # Choose random node
        node_i = np.random.choice(list(self.network.nodes()))

        # Update potential
        potential = self.get_neuron_potential(node_i)
        neighbors = self.get_neighbors(node_i)

        new_potential = potential + len(neighbors)

        # Check if threshold is exceeded
        if new_potential >= len(neighbors):
            new_potential -= len(neighbors)

            # If yes, redistribute potential
            for node_j in neighbors:
                self.add_neuron_potential(node_j, 1.0)

        # Repeat threshold check for every affected node

    def run(self):
        for i in range(10):
            self.step()


if __name__ == '__main__':
    model = NeuronModel(10)

    print(model.network)

    model.run()
