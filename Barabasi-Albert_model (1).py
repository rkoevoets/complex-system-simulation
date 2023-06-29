#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from main_numpy import NeuronModel



if __name__ == '__main__':
    ba_network = nx.barabasi_albert_graph(400, 5, seed=123, initial_graph=None)
    ba_network.add_node(-1)  # Adding the sink node

    p = 0.5  # Probability of adding an edge to the sink

# Iterate over each node (not the sink node)
    for node in ba_network.nodes():
        if node != -1:
            if np.random.random() < p:
                ba_network.add_edge(node, -1)

    model = NeuronModel(ba_network, sample_delay=200, start_filled=False)
    data = np.array(model.run(100000))
    sizes, counts = np.unique(data, return_counts=True)
    
##Plots:    

#1. Avalanche size time series 
    plt.title("Avalanche time series")
    plt.plot(data)
    plt.xlabel("step")
    plt.ylabel("data")
    #plt.savefig("")
    plt.show()
    
#2. Avalanche size distribution for m = 2 and N = 400
    plt.title("m=2 and N=400")
    plt.scatter(sizes,counts, color='red')  # Convert back to original scale
    
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("s")
    plt.ylabel("P(s)")
    plt.savefig("BA")
    plt.show()


# In[17]:


# Define the model function
def pl_with_exp_cutoff(x, a, b, c):
    return a * (x**-b) * np.exp(-c * x)

p = 0.2  # Probability of adding an edge to the sink



#3. Avalanche size distribution for varying parameter m ()
if __name__ == '__main__':
    m = [2,5,10]
    N = 1000
    plt.title("N="+str(N))
    for i in range(len(m)): #go through the list to get different m
        network = nx.barabasi_albert_graph(N, m[i], seed=123, initial_graph=None)
        network.add_node(-1)  # Adding the sink node

        # Iterate over each node (not the sink node)
        for node in network.nodes():
            if node != -1:
                if np.random.random() < p:
                    network.add_edge(node, -1)

        model = NeuronModel(network, sample_delay=1000, start_filled=False)
        data = np.array(model.run(50000))
        sizes, counts = np.unique(data, return_counts=True)

        #Plot avalanche distribution for specific N
        plt.scatter(sizes, counts,label="m="+str(m[i]))#label=r'$\tau$='+str(round(param[0], 2)))
        plt.xscale("log") 
        plt.yscale("log")
        plt.xlabel("s")
        plt.ylabel("Frequency")
        plt.legend()
    plt.savefig("BA_frequency_m")
    plt.show()

#3. Avalanche size distribution for varying parameter m ()
if __name__ == '__main__':
    m = [2,5,10]
    N = 1000
    plt.title("N="+str(N))
    for i in range(len(m)): #go through the list to get different m
        network = nx.barabasi_albert_graph(N, m[i], seed=123, initial_graph=None)
        network.add_node(-1)  # Adding the sink node


        # Iterate over each node (not the sink node)
        for node in network.nodes():
            if node != -1:
                if np.random.random() < p:
                    network.add_edge(node, -1)

        model = NeuronModel(network, sample_delay=1000, start_filled=False)
        data = np.array(model.run(50000))
        sizes, counts = np.unique(data, return_counts=True)
        
        # Perform curve fitting
        print("avalanche_sizes=",sizes[sizes<200])
        popt, pcov = curve_fit(pl_with_exp_cutoff, sizes[sizes<200], counts[sizes<200]/sum(counts))
        
        # Generate fitted curve values
        x_fit = np.logspace(np.log10(min(sizes)), np.log10(max(sizes)), 100)
        y_fit = pl_with_exp_cutoff(x_fit, *popt)
        norm_constant = np.trapz(y_fit, x=x_fit)

        #Plot avalanche distribution for specific N
        plt.plot(x_fit, y_fit / norm_constant,label="m="+str(m[i]))#label=r'$\tau$='+str(round(param[0], 2)))
        plt.xscale("log") 
        plt.yscale("log")
        plt.xlabel("s")
        plt.ylabel("P(s,m)")
        plt.legend()
    plt.savefig("BA_p_m")
    plt.show()


# In[8]:


# Define the model function
def pl_with_exp_cutoff(x, a, b, c):
    return a * x**b * np.exp(-c * x)


#4. Avalanche size distribution for varying Node numbers ()
if __name__ == '__main__':
    N = [100,400,1600,3600] #Keep in mind the problem with N=5000
    m = 2
    plt.title("m="+str(m))
    for i in range(len(N)): #go through the list to get different N
        
        #Extract Data
        network = nx.barabasi_albert_graph(N[i],m, seed=123, initial_graph=None)
        network.add_node(-1)  # Adding the sink node

        p = 0.4  # Probability of adding an edge to the sink

        # Iterate over each node (not the sink node)
        for node in network.nodes():
            if node != -1:
                if np.random.random() < p:
                    network.add_edge(node, -1)

        model = NeuronModel(network, sample_delay=10000, start_filled=False)
        data = np.array(model.run(100000))
        sizes, counts = np.unique(data, return_counts=True)
    
        
        # Perform curve fitting
        popt, pcov = curve_fit(pl_with_exp_cutoff, sizes[0:150], counts[0:150])
        
        # Generate fitted curve values
        x_fit = np.logspace(np.log10(min(sizes)), np.log10(max(sizes)), 100)
        y_fit = pl_with_exp_cutoff(x_fit, *popt)
        norm_constant = np.trapz(y_fit, x=x_fit)

        #Plot avalanche distribution for specific N
        #plt.scatter(sizes, counts/sum(counts), marker='.',label="N="+str(N[i]))
        plt.plot(x_fit, y_fit / norm_constant,label="N="+str(N[i]))#label=r'$\tau$='+str(round(param[0], 2)))
        plt.xscale("log") 
        plt.yscale("log")
        plt.xlabel("s")
        plt.ylabel("P(s,N)")
        plt.legend()
    plt.savefig("BA_N.png", dpi = 300)
    plt.show()
    
#5. Avalanche size distribution with data collapse
    plt.title("Data collapse")
    for i in range(len(N)):
        
        #Extract Data
        network = nx.barabasi_albert_graph(N[i],m, seed=123, initial_graph=None)
        network.add_node(-1)  # Adding the sink node

        p = 0.5  # Probability of adding an edge to the sink

        # Iterate over each node (not the sink node)
        for node in network.nodes():
            if node != -1:
                if np.random.random() < p:
                    network.add_edge(node, -1)

        model = NeuronModel(network, sample_delay=10000, start_filled=False)
        data = np.array(model.run(100000))
        sizes, counts = np.unique(data, return_counts=True)
        
        # Perform curve fitting
        popt, pcov = curve_fit(pl_with_exp_cutoff, sizes, counts)
        
        # Generate fitted curve values
        x_fit = np.logspace(np.log10(min(sizes)), np.log10(max(sizes)), sizes.size)
        y_fit = pl_with_exp_cutoff(x_fit, *popt)
        norm_constant = np.trapz(y_fit, x=x_fit)
        
        plt.plot(x_fit/np.max(x_fit),(x_fit**-popt[1])*(y_fit/norm_constant),label="N="+str(N[i]))
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel(r"s/$s_c$")
        plt.ylabel(r"s$^\tau$ P(s,N)")
        plt.legend()
    plt.savefig("Data collapse.png",dpi=300)
    plt.show()


# In[ ]:




