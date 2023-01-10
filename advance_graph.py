from collections import Counter
from sympy import symbols, simplify, Pow
from prefixed import Float
from lcapy import Circuit

from find_trees import find_all_spanning_trees_mp, find_all_spanning_trees, find_all_two_trees

s = symbols('s')


class AdvanceGraph(object):
    def __init__(self, graph, edge_metadata):
        """
        Stores the metadata of each edge in a graph. The metadata includes the component type and the value.
        Args:
            graph: networkx.Graph
                the networkx graph
            edge_metadata: dict
                the metadata of components
        """
        self.graph = graph
        self.metadata = edge_metadata

    def compute_tree_product(self, tree: list):
        """only support R, C, L"""
        total_sum = 0
        for tree_graph in tree:
            edges = list(tree_graph.edges())
            value_of_edges = []
            for edge in edges:
                value_of_edges.append(self._find_component_value(edge))
            combinations = list(self._product([iter(x) for x in value_of_edges]))
            for indice in combinations:
                pr, po = 1, 0
                for element in indice:
                    pr *= element[0]
                    po += element[1]
                total_sum += pr * Pow(s, po)
        return simplify(total_sum)

    def find_single_tree(self):
        return find_all_spanning_trees(self.graph)

    def find_two_tree(self, tree, excluded_pair):
        return find_all_two_trees(self.graph, tree, excluded_pair)

    def _find_component_value(self, edge: tuple):
        s = {'Resistor': 0, 'BehavioralCapacitor': 1, 'BehavioralInductor': -1}
        component_value = []
        for k in self.metadata:
            m = self.metadata[k]
            if Counter(m[-1]) == Counter(edge):
                component_value.append([Float(m[1]), s[m[0]]])
        return component_value

    def _product(self, args, repeat=1):
        # product('ABCD', 'xy') --> Ax Ay Bx By Cx Cy Dx Dy
        # product(range(2), repeat=3) --> 000 001 010 011 100 101 110 111
        pools = [tuple(pool) for pool in args] * repeat
        result = [[]]
        for pool in pools:
            result = [x+[y] for x in result for y in pool]
        for prod in result:
            yield tuple(prod)


class AdvanceCircuit(Circuit):
    def __init__(self, file_name):
        super(AdvanceCircuit, self).__init__(filename=file_name)
        """
        Inherit from the Circuit class
        Args:
            file_name
                address of the netlist file
        """
        self.networkx_graph = self.circut_graph().G

    def compute_tree_product(self, tree: list):
        """only support R, C, L"""
        total_sum = 0
        for tree_graph in tree:
            edges = list(tree_graph.edges())
            value_of_edges = []
            for edge in edges:
                value_of_edges.append(self._find_component_value(edge))
            combinations = list(self._product([iter(x) for x in value_of_edges]))
            for indice in combinations:
                pr, po = 1, 0
                for element in indice:
                    pr *= element[0]
                    po += element[1]
                total_sum += pr * Pow(s, po)
        return simplify(total_sum)

    def find_single_tree(self):
        return find_all_spanning_trees(self.graph)

    def find_two_tree(self, tree, excluded_pair):
        return find_all_two_trees(self.graph, tree, excluded_pair)

    def _find_component_value(self, edge: tuple):
        s = {'Resistor': 0, 'BehavioralCapacitor': 1, 'BehavioralInductor': -1}
        component_value = []
        for k in self.metadata:
            m = self.metadata[k]
            if Counter(m[-1]) == Counter(edge):
                component_value.append([Float(m[1]), s[m[0]]])
        return component_value

    def _product(self, args, repeat=1):
        # product('ABCD', 'xy') --> Ax Ay Bx By Cx Cy Dx Dy
        # product(range(2), repeat=3) --> 000 001 010 011 100 101 110 111
        pools = [tuple(pool) for pool in args] * repeat
        result = [[]]
        for pool in pools:
            result = [x+[y] for x in result for y in pool]
        for prod in result:
            yield tuple(prod)








