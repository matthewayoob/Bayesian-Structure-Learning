import sys
import numpy as np
import networkx as nx
import pandas as pd
import io
import scipy.stats
import tempfile
import graphviz
import os
import matplotlib.pyplot as plt
import pylab as plt
import random 
import tqdm
import time

def print_network(network):
    print("Nodes in the DiGraph:")
    for node in network.nodes():
        print(node)
    print("\nEdges in the DiGraph:")
    for edge in network.edges():
        print(edge)

# names2index = {v: k for k, v in idx2names.items()}

#Done: Read gph, print network, write GPH, draw graph, bayesian component, bayesian score,  k2, score, compute, main

# Create new class to represent a variable - chapter 1 
class Variable():
    """
    A variable is given a name (represented as a string) and may take on an integer from 0 to r - 1
    """
    def __init__(self, name: str, numValues: int):
        self.name = name
        self.numValues = numValues  # possible values amt /r

    def __print__(self):
        return "(Index Representation:" + str(self.name) + " ; n: " + str(self.numValues) + ")"
    
def initial_work(csv_name):
    df = pd.read_csv(csv_name)
    node_names = df.columns 
    nmdp = df.to_numpy()
    maxr = np.max(nmdp, axis=0) # grab the max of every column

    vars = []
    i = 0
    for row in df:
        vars.append(Variable(i, max(df[row])))
        i += 1
    idx2names = {}
    G = nx.DiGraph()
    node_amt = len(list(vars)) 
    print(node_amt)
    for node in range(node_amt):
        G.add_node(node)
        idx2names[node] = node_names[node]
    return df, node_names, node_amt, nmdp, maxr, G, idx2names

# vars = [Variable('0', 3), Variable('1', 3), Variable('2', 3), Variable('3', 3), Variable('4', 3), Variable('5', 3)]

# take prior, statistics - chapter 2  
def stats(G: nx.DiGraph, node_amt, nmdp, maxr):
    # Check the parent count
    q = [int(np.prod([maxr[j] for j in G.predecessors(i)])) for i in range(node_amt)] 

    # Initialize arrays for statistics and prior
    M = [np.zeros((q[i], maxr[i])) for i in range(node_amt)]

    # Iterate over the data
    for o in nmdp:
        for i in range(node_amt): 
            k = o[i]
            par = [p for p in G.predecessors(i)]
            j = 0 

            if par:  # Check if there are any parents
                j = np.unravel_index(o[par] - 1, maxr[par])

            M[i][j, k - 1] += 1.0
    return M

def prior(graph, M, node_amt) -> list[np.ndarray]:
    pri = [np.ones_like(M[i]) for i in range(node_amt)]
    # prior = [np.ones((q[i], maxr[i])) for i in range(node_amt)]
    return pri

# take bayesian component, take bayesian score - chapter 5

def bayesian_score(G, node_amt, nmdp, maxr): 
    def bayesian_score_component(M, a): 
        p = np.sum(scipy.special.loggamma(a + M))
        p -= np.sum(scipy.special.loggamma(a))
        p -= np.sum(scipy.special.loggamma(np.sum(a, axis=1) + np.sum(M, axis=1)))
        p += np.sum(scipy.special.loggamma(np.sum(a, axis=1)))
        return p
    M = stats(G, node_amt, nmdp, maxr)
    alpha =  prior(G, M, node_amt)
    return np.sum([bayesian_score_component(M[i], alpha[i]) for i in range(node_amt)])  


def k2(G, nmdp, node_amt, maxr):
    # Create a random order of nodes
    order = random.sample(range(node_amt), node_amt)
    
    for i in tqdm.tqdm(range(1, node_amt)):
        node = order[i]
        y = bayesian_score(G, node_amt, nmdp, maxr)

        while True:
            y_best = -float("inf")
            test_node_best = 0
            
            # Iterate through previously added nodes
            for j in range(0, i):
                test_node = order[j]
                
                # Check if there is no edge between test_node and the current node
                if not G.has_edge(test_node, node):
                    G.add_edge(test_node, node)
                    y_prime = bayesian_score(G, node_amt, nmdp, maxr)

                    # Update the best score and the corresponding test node
                    if y_prime > y_best:
                        y_best, test_node_best = y_prime, test_node
                    
                    # Remove the edge for further evaluation
                    G.remove_edge(test_node, node)
            
            # If a better score is found, add the edge to the graph
            if y_best > y:
                y = y_best
                G.add_edge(test_node_best, node)
            else:
                break
    
    return G

# write compute which takes in a file and computes the bayesian score 
# to test run - python3 project1,py example.csv example.gph - sub in .csv and .gph for the dataset you are working with 
# download the example - run only with the example in the beginning, get data thorugh read gph
# test with statistics first 
# big thing to watch out for: data is 1 indexed - so when you are reading it in you have to think about how you wanna manipulate 

def write_gph(nw, filename, idx2names):
    edges = []
    for key, values in nx.to_dict_of_lists(nw).items():
        for value in values:
            edges.append('{},{}\n'.format(key, value))
    with open(filename, 'w') as f:
        f.writelines(edges)

def draw_graph(network, filename):
    layout = nx.spring_layout(network)
    nx.draw(network, pos=layout, with_labels=True, node_size=500, node_color="skyblue", arrows=True)
    plt.savefig(filename, format="png")
    plt.show()

def read_gph(G, filename, idx2names):
    with open(filename, 'w') as f:
        for edge in G.edges():
            f.write("{},{}\n".format(idx2names[edge[0]], idx2names[edge[1]]))

def compute(infile):
    output = str(infile)[:-4] + ".gph"
    csv_name = infile
    df, node_names, node_amt, nmdp, maxr, G, idx2names = initial_work(csv_name)
    G_new = k2(G, nmdp, node_amt, maxr)
    print("Bayesian Score:", bayesian_score(G_new, node_amt, nmdp, maxr))
    write_gph(G_new, output, idx2names)
    read_gph(G_new, output, idx2names)
    # draw_graph(G_new, output)

    # file_path = "small.gph"

    # network = nx.DiGraph()
    # network.add_edge("fixedacidity","volatileacidity")
    # network.add_edge("fixedacidity", "residualsugar")
    # network.add_edge("citricacid", "residualsugar")
    # network.add_edge("chlorides", "residualsugar")
    # network.add_edge("chlorides", "freesulfurdioxide")
    # draw_graph(network, output)

    # network = nx.DiGraph()
    # network.add_edge("HW", "FA")    
    # network.add_edge("EF", "FA")
    # network.add_edge("QV", "FA")
    # network.add_edge("QV", "PZ")

    # network = nx.DiGraph()
    # network.add_edge("age", "portembarked")    
    # network.add_edge("age", "numparentschildren")
    # network.add_edge("fare", "numparentschildren")
    # network.add_edge("passengerclass", "numparentschildren")
    # network.add_edge("passengerclass", "sex")
    # draw_graph(network, output)

    # making sample network
    # network = nx.DiGraph()
    # network.add_edge(0, 1)
    # network.add_edge(0, 3)    
    # network.add_edge(2, 3)
    # network.add_edge(4, 3)
    # network.add_edge(4, 5)

# def display_graph():
#     small = [
#         ['age', 'numsiblings'],
#         ['portembarked', 'sex'],
#         ['portembarked', 'passengerclass'],
#         ['passengerclass', 'fare'],
#         ['sex', 'survived'],
#         ['numsiblings', 'portembarked'],
#         ['numsiblings', 'numparentschildren'],
#     ]
    
#     medium = [
#     ['fixedacidity', 'ph'],
#     ['fixedacidity', 'citricacid'],
#     ['volatileacidity', 'alcohol'],
#     ['residualsugar', 'density'],
#     ['alcohol', 'quality'],
#     ['alcohol', 'sulphates'],
#     ['alcohol', 'residualsugar'],
#     ['alcohol', 'totalsulfurdioxide'],
#     ]
#     large = [
#     ["HW", "AG"],
#     ["EF", "ZD"],
#     ["EF", "LD"],
#     ["EF", "WZ"],
#     ["QV", "EF"],
#     ["QV", "GZ"],
#     ["QV", "JJ"],
#     ["QV", "CH"],
#     ["QV", "SM"],
#     ["QV", "EV"],
#     ["PZ", "LP"],
#     ["PZ", "YF"],
#     ["PZ", "FA"],
#     ["WA", "PI"],
#     ["WA", "QV"],
#     ["JJ", "VJ"],
#     ["JJ", "KW"],
#     ["RA", "SI"],
#     ["BN", "YU"],
#     ["FL", "IO"],
#     ["FL", "NV"],
#     ["YU", "HY"],
#     ["NV", "ST"],
#     ["VJ", "PQ"],
#     ["CH", "GO"],
#     ["CH", "SQ"],
#     ["CH", "MD"],
#     ["TN", "FH"],
#     ["JD", "LO"],
#     ["FH", "VX"],
#     ["EM", "FL"],
#     ["QJ", "BN"],
#     ["SI", "BY"],
#     ["SI", "EJ"],
#     ["SA", "GI"],
#     ["PT", "WA"],
#     ["PT", "RA"],
#     ["PT", "PZ"],
#     ["SB", "ZY"],
#     ["SB", "KO"],
#     ["SB", "HW"],
#     ["EN", "QJ"],
#     ["EN", "CO"],
#     ["EN", "JD"],
#     ["PI", "TN"],
# ]
#     G = nx.DiGraph()
#     for line in large:
#         G.add_edge(line[0], line[1])
#         print(line[0], line[1])
#     draw_graph(G, "Large graph") 
    return
    
def main():
    if len(sys.argv) != 2:
        raise Exception("usage: python project1.py <infile>.csv -> <infile>.gph")

    inputfilename = sys.argv[1]
    start_time = time.time()
    compute(inputfilename)
    elapsed_time = time.time() - start_time
    print("Total runtime:", elapsed_time)
    return

if __name__ == '__main__':
    main()