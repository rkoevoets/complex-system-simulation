import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.stats import linregress
from main_numpy import NeuronModel


network = nx.erdos_renyi_graph(100, 0.5, directed=False)
network.add_node(-1)
network.add_edge(0, -1)

model = NeuronModel(network)

data = model.run(50000)

# plotting histogram
plt.hist(data, density=True, log=True)

plt.title("Avalanche size distribution")
plt.xlabel("s")
plt.ylabel("P(s)")

plt.show()

# plotting scatter plot
avel, freq = np.unique(data, return_counts=True)

plt.title('Avelanche frequenties')
plt.xlabel('s')
plt.ylabel('P(s)')
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
plt.ylabel('s')
plt.plot(x_axis_time_series, y_axis_time_series)
plt.show()


def multiple_p_values():
    # initializing empty lists
    data_list = []
    clustering_list = []
    avelanche_list = []
    frequency_list = []
    change_p_value_list = []

    # creating the values for the multiple p graph
    for change_p_value in np.arange(0.2, 1, 0.2):
        network = nx.erdos_renyi_graph(100, change_p_value, directed=False)
        network.add_node(-1)
        network.add_edge(0, -1)

        model = NeuronModel(network)
        data = np.array(model.run(50000), dtype=object)
        avel, freq = np.unique(data, return_counts=True)
        change_p_value_list.append(change_p_value)

        # appending values to list
        avelanche_list.append(avel)
        frequency_list.append(freq)
        data_list.append(data)

    # plotting the scatterplot
    for i in range(len(avelanche_list)):
        plt.scatter(avelanche_list[i], frequency_list[i], label=f'p value {change_p_value_list[i]}')

    plt.title('Avalanche size frequency')
    plt.xlabel('s')
    plt.ylabel('P(s)')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.show()


multiple_p_values()


def different_N_values():
    # initializing empty lists
    nodes_list = []
    freq_list = []
    n_value_list = [400, 1600, 10000]
    change_N_value_list = []

    # creating the values for the multiple N graph
    for change_N_value in n_value_list:
        network = nx.erdos_renyi_graph(change_N_value, 0.5, directed=False)
        network.add_node(-1)
        network.add_edge(0, -1)

        model = NeuronModel(network)
        data = np.array(model.run(50000), dtype=object)
        nodes, freq = np.unique(data, return_counts=True)
        change_N_value_list.append(change_N_value)

        # appending values to list
        nodes_list.append(nodes)
        freq_list.append(freq)

    # plotting the scatterplot
    for i in range(len(nodes_list)):
        plt.scatter(freq_list[i], nodes_list[i], label=f'N value {change_N_value_list[i]}')

    plt.title('Avalanche size  in comparison to amount of nodes int the network')
    plt.xlabel('s')
    plt.ylabel('P(s)')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.show()


different_N_values()