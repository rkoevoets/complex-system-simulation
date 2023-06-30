import networkx as nx
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from main_numpy import NeuronModel
import matplotlib

'Plotting ER network'

network = nx.erdos_renyi_graph(20, 0.1, directed=False)
network.add_node(-1)  # Adding the sink node

p = 0.1  # Probability of adding an edge to the sink

# Iterate over each node (not the sink node)
for node in network.nodes():
    if node != -1:
        if np.random.random() < p:
            network.add_edge(node, -1)

nx.draw_networkx(network, pos=nx.spring_layout(network))
plt.title('ER graph of N = 20, p = 0.1')
plt.show()

'Plotting BA'

ba_network = nx.barabasi_albert_graph(20, 2, seed=None, initial_graph=None)
ba_network.add_node(-1)  # Adding the sink node

# Iterate over each node (not the sink node)
for node in ba_network.nodes():
    if node != -1:
        if np.random.random() < p:
            ba_network.add_edge(node, -1)
nx.draw_networkx(ba_network, pos=nx.spring_layout(network))
plt.title('BA graph of N = 20, m = 2')
plt.show()

'Plotting WS'

ws_network = nx.watts_strogatz_graph(20, k=2, p=0.5, seed=None)
ws_network.add_node(-1)  # Adding the sink node

# Iterate over each node (not the sink node)
for node in ws_network.nodes():
    if node != -1:
        if np.random.random() < p:
            ws_network.add_edge(node, -1)
nx.draw_networkx(ws_network, pos=nx.spring_layout(network))
plt.title('WS graph of N = 20, k = 2, p = 0.5')
plt.show()
