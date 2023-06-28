import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.stats import linregress


class NeuronModel:
    """Class containing attributes and methods for the sand-pile model.
    """
    def __init__(self, size, p_value, sample_delay=200) -> None:
        self.network = nx.erdos_renyi_graph(size, p_value, directed=False)

        # Add sink node connections (boundaries)
        self.network.add_node(-1)

        for comp in nx.connected_components(self.network):
            self.network.add_edge(-1, list(comp)[0])

        assert nx.is_connected(self.network), "All nodes must be able to reach a sink node"

        self.size = size  # Without the sink node

        self.adj_matrix = nx.adjacency_matrix(self.network).toarray()
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

        return np.array(self.avalanche_sizes), nx.average_clustering(self.network)


if __name__ == '__main__':
    model = NeuronModel(10, 0.5)
    # neuron model data
    data, clustering = np.array(model.run(10000), dtype = object)
    # data = data[1000:]
    print(f'The clustering coefficient of the model is {clustering}')

    # plotting histogram
    plt.hist(data, density=True, log=True)

    plt.title("Avalanche size distribution")
    plt.xlabel("s")
    plt.ylabel("P(s)")
    plt.show()

    # plotting scatter plot
    avel, freq = np.unique(data, return_counts=True)

    plt.title('Avelanche frequenties')
    plt.xlabel('Amount of avelanches')
    plt.ylabel('Frequency of the avelanche')
    plt.xscale('log')
    plt.yscale('log')
    plt.scatter(avel, freq)
    plt.show()

    # linear regression
    log_avel = np.log10(avel)
    log_freq = np.log10(freq)

    slope, intercept, r_value, p_value, std_err = linregress(log_avel, log_freq)
    print(f'The value of the slope is {slope}')

    # calculating clustering coefficient of model (doesn't work yet)
    # clust_coef = nx.average_clustering(model)

    # avalanche size as function of time series
    x_axis_time_series = [i for i in range(len(model.avalanche_sizes))]
    y_axis_time_series = model.avalanche_sizes
    plt.title('Time series avalanche sizes')
    plt.xlabel('Time')
    plt.ylabel('Avalanche sizes')
    plt.plot(x_axis_time_series,y_axis_time_series)
    plt.show()

    def multiple_p_values():

        #initializing empty lists
        data_list = []
        clustering_list = []
        avelanche_list = []
        frequency_list = []
        change_p_value_list = []

        # creating the values for the multiple p graph
        for change_p_value in np.arange(0.2, 1, 0.2):
            model = NeuronModel(10, change_p_value)
            data, clustering = np.array(model.run(10000), dtype=object)
            avel, freq = np.unique(data, return_counts=True)
            change_p_value_list.append(change_p_value)

            # appending values to list
            avelanche_list.append(avel)
            frequency_list.append(freq)
            data_list.append(data)
            clustering_list.append(clustering)

        #plotting the scatterplot
        for i in range(len(avelanche_list)):
            plt.scatter(avelanche_list[i],frequency_list[i], label=f'p value {change_p_value_list[i]}')

        plt.title('Avalanche size frequency')
        plt.xlabel('Avalanche size')
        plt.ylabel('Frequency')
        plt.xscale('log')
        plt.yscale('log')
        plt.legend()
        plt.show()


    multiple_p_values()

    def different_N_values():

        #initializing empty lists
        nodes_list = []
        freq_list = []
        change_N_value_list = []

        # creating the values for the multiple N graph
        for change_N_value in np.arange(10, 100, 20):
            model = NeuronModel(change_N_value, 0.5)
            data, clustering = np.array(model.run(10000), dtype=object)
            nodes, freq = np.unique(data, return_counts=True)
            change_N_value_list.append(change_N_value)

            # appending values to list
            nodes_list.append(nodes)
            freq_list.append(freq)

        # plotting the scatterplot
        for i in range(len(nodes_list)):
            plt.scatter(freq_list[i], nodes_list[i], label=f'N value {change_N_value_list[i]}')


        plt.title('Avalanche size  in comparison to amount of nodes int the network')
        plt.xlabel('Avalanche size')
        plt.ylabel('Frequency')
        plt.xscale('log')
        plt.yscale('log')
        plt.legend()
        plt.show()




    different_N_values()













