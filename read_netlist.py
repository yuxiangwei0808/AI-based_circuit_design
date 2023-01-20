import helpers as h
from os import listdir
from os.path import isfile, join
import time
import networkx as nx
import numpy as np
from find_trees import find_all_spanning_trees

path = "./data/ltspice_examples/"
files = [path+f for f in listdir(path) if isfile(join(path, f))]
files = ['./raw_netlist/figure26.net']
netlists = ((f, open(f, 'rb').read().decode('utf-8', 'ignore')) for f in files)

valid_files = [(f, src) for (f, src) in netlists if h.is_valid_netlist(src)]
graph_data = {}
index = 0
for (f, src) in valid_files:
        # print(f)
        component_list, g = h.netlist_to_graph(src)
        t = time.time()
        tree = [x for x in g.find_single_tree()]
        print(time.time() - t) # 94
        two_tree = g.find_two_tree(tree, [[1, 6], [4, 2]])
#         graph_data[index] = (component_list, g)
#         index +=1
#
# dataset = {}
# print("len graph data:", len(graph_data))
# for ind, (c,g) in graph_data.items():
#         for i , component in enumerate(c):
#                 g_ = g.copy()
#                 new_g = g_.remove_node(i)
#                 if component in dataset.keys():
#                         dataset[component].extend(new_g)
#                 else:
#                         dataset[component] = [new_g]

