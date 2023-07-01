#!/usr/bin/env python
# coding: utf-8

# In[12]:


import networkx as nx
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from main_numpy import NeuronModel
import matplotlib

'''Change p and p vor the edges to change edge connectivity chance and edge sink connectivity chance'''
if __name__ == '__main__':
    colors = ['blue', 'green', 'orange', 'red' ]
    N = [100,400,1600,3600] #Keep in mind the problem with N=5000
    for i in range(len(N)): #go through the list to get different N

        #Extract Data
        network = nx.erdos_renyi_graph(N[i], 0.1, directed=False)
        network.add_node(-1)  # Adding the sink node

        p = 0.9  # Probability of adding an edge to the sink

        # Iterate over each node (not the sink node)
        for node in network.nodes():
            if node != -1:
                if np.random.random() < p:
                    network.add_edge(node, -1)

        model = NeuronModel(network, sample_delay=10000, start_filled=True)
        data = np.array(model.run(100000))
        sizes, counts = np.unique(data, return_counts=True)


        #Plot avalanche distribution for specific N
        plt.scatter(sizes, counts, marker='.',label="N="+str(N[i]), color= colors[i], s = 120)
        plt.xscale("log")
        plt.yscale("log")

        plt.title('Avalanche size distribution with p = 0.1 and p_sink = 0.9', fontsize = '15')
        plt.xlabel("s")
        plt.ylabel("frequency")
        plt.legend()
        plt.savefig("ER_frequency.png", dpi = 300)
        plt.show()

########################
