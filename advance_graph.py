from collections import Counter
from sympy import symbols, simplify, Pow
from prefixed import Float
from lcapy import Circuit, sexpr, impedance

from find_trees import find_all_spanning_trees_mp, find_all_spanning_trees, find_all_two_trees
from utils import *

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
        self.networkx_graph = self.circuit_graph().G.copy()
        self._remove_dummy_node()

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
                tmp = 1
                for i in indice:
                    tmp *= i.as_expr()
                total_sum += tmp
                # pr, po = 1, 0
                # for element in indice:
                #     pr *= element[0]
                #     po += element[1]
                # total_sum += pr * Pow(s, po)
        return total_sum

    def find_single_tree(self):
        return find_all_spanning_trees(self.networkx_graph)

    def find_two_tree(self, tree, excluded_pair):
        return find_all_two_trees(self.networkx_graph, tree, excluded_pair)

    def _find_component_value(self, edge: tuple):
        component_value = []
        for k in self._component_metadata:
            m = self._component_metadata[k]
            if m[1] != 'V':
                if Counter(m[0]) == Counter(edge):
                    if m[1] == 'R':
                        name = str(m[-1])
                    elif m[1] == 'C':
                        name = str(m[-1]) + '/s'
                    elif m[1] == 'L':
                        name = str(m[-1]) + '*s'
                    else:
                        raise NotImplementedError
                    component_value.append(impedance(sexpr(name)))
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

    def _connect_two_nodes(self, node_pair: tuple, component_metadata: tuple):
        self = connect_two_node(self, node_pair, component_metadata)
        self._update_networkx()

    def _insert_new_node(self, node_pair: tuple, component_metadata: tuple):
        self = insert_new_node(self, node_pair, component_metadata)
        self._update_networkx()

    def _remove_dummy_node(self):
        for node in list(self.circuit_graph().nodes):
            if '*' in node:
                self.networkx_graph.remove_node(node)

    def _find_component_by_nodes(self, node_pair: tuple):
        for c in self.elements.keys():
            component = self.elements[c]
            if not component.is_voltage_source:
                if set(component.relnodes) == set(node_pair):
                    return component.relnodes, component.relname, component
        print('no component between the nodes')
        raise Exception

    def _update_networkx(self):
        self.networkx_graph = self.circuit_graph().G.copy()
        self._remove_dummy_node()

    @property
    def _component_metadata(self):
        """use dict to store all components' metadata"""
        metadata = {}
        for c in self.elements.keys():
            if len(self.elements[c].nodenames) > 1:
                component = self.elements[c]
                nodes = component.relnodes
                name, c_type = component.name, component.type
                value = get_value(component)
                if component in metadata.keys():
                    raise Exception
                metadata[c] = [nodes, c_type, value]

        return metadata


if __name__ == '__main__':
    file_name = './raw_netlist/figure30.net'
    advance_circuit = AdvanceCircuit(file_name)
    advance_circuit._insert_new_node(('N001', 'N002'), ('R4', 1.6))
    tree = advance_circuit.find_single_tree()
    value = advance_circuit.compute_tree_product(tree)
    print(1)




