from collections import Counter
from sympy import symbols, simplify, Pow
from prefixed import Float
from lcapy import Circuit, sexpr, admittance
import copy
import networkx as nx
import matplotlib.pyplot as plt

from find_trees import find_all_spanning_trees, find_all_two_trees
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

        self.single_tree = None
        self.two_tree = {}

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

    def find_two_tree(self, excluded_pair):
        if self.single_tree is None:
            self.single_tree = self._find_single_tree()
        if str(sorted(excluded_pair)) in self.two_tree.keys():
            return self.two_tree[str(sorted(excluded_pair))]
        else:
            tree = copy.deepcopy(self.single_tree)
            t = find_all_two_trees(self.networkx_graph, tree, excluded_pair)
            self.two_tree[str(sorted(excluded_pair))] = t
            return t

    def deepcopy(self):
        new = self.copy()
        new = AdvanceCircuit(str(new))
        return new

    def connect_two_nodes(self, node_pair: list, component_metadata: list):
        self = connect_two_node(self, node_pair, component_metadata)
        self._update_networkx()

    def connect_three_nodes(self, nodes: list, name: str):
        self = connect_three_nodes(self, nodes, name)
        self._update_networkx()

    def insert_new_node(self, node_pair: list, component_metadata: list):
        self = insert_new_node(self, node_pair, component_metadata)
        self._update_networkx()

    def insert_new_node_bjt(self, nodes: list, bjt_name: str):
        self = insert_new_node_bjt(self, nodes, bjt_name)
        self._update_networkx()

    def insert_new_node_to_bjt(self, nodes: list, component_name: str, bjt_name: str, new_component_value: str = None):
        self = insert_new_node_to_bjt(self, nodes, component_name, bjt_name, new_component_value)
        self._update_networkx()

    def change_element_connection(self, element, new_pair):
        if element[0] == 'Q':
            assert len(new_pair) == 3
            self = change_element_connection(self, element, new_pair)
        else:
            value = self._component_metadata[element][-1]
            self = change_element_connection(self, element, new_pair, value)

        self = connect_standalone_node_to_ground(self, in_out_nodes=['NC_01', 'out'])
        self._update_networkx()

    def alter_component_value(self, component, value):
        nodes = self._component_metadata[component][0]
        ist = [component, nodes[0], nodes[1], str(value)]
        ist = ' '.join(ist)
        self.add(ist)

    def delete_component(self, component: str):
        self.remove(component)
        self = connect_standalone_node_to_ground(self, in_out_nodes=['NC_01', 'out'])
        self._update_networkx()

    def _find_component_value(self, edge: tuple):
        # find component admittance
        component_value = []
        pair = str(sorted(edge))
        names = self._edge_nodes_metadata[pair]
        for name in names:
            m = self._component_metadata[name]
            if m[1] == 'R':
                name = '1/(' + str(m[-1]) + ')'
            elif m[1] == 'C':
                name = str(m[-1]) + '*s'
            elif m[1] == 'L':
                name = '1/(' + str(m[-1]) + ')/s'
            else:
                raise NotImplementedError
            component_value.append(admittance(sexpr(name)))
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

    def _find_single_tree(self):
        return find_all_spanning_trees(self.networkx_graph)

    def _remove_dummy_node(self):
        for node in list(self.circuit_graph().nodes):
            if '*' in node:
                self.networkx_graph.remove_node(node)

    def _find_component_by_nodes(self, node_pair: tuple):
        for c in self.elements.keys():
            component = self.elements[c]
            if set(component.relnodes) == set(node_pair):
                return component.relnodes, component.relname, component
        print('no component between the nodes')
        raise Exception

    def _update_networkx(self):
        self.networkx_graph = self.circuit_graph().G.copy()
        self._remove_dummy_node()

    @property
    def _component_metadata(self):
        """use dict to store all components' metadata, currently exclude source"""
        metadata = {}
        for c in self.elements.keys():
            if len(self.elements[c].nodenames) > 1 and c[0] != 'Q':
                component = self.elements[c]
                nodes = component.relnodes
                name, c_type = component.name, component.type
                assert c_type != 'V'
                value = get_value(component)
                if component in metadata.keys():
                    raise Exception
                metadata[c] = [nodes, c_type, value]
        return metadata

    @property
    def _edge_nodes_metadata(self):
        """return the nodes and the corresponding edge (component)"""
        metadata = {}
        for c in self.elements:
            if len(self.elements[c].nodenames) > 1:
                component = self.elements[c]
                nodes = sorted(component.nodenames)
                if str(nodes) in metadata:
                    metadata[str(nodes)].append(c)
                else:
                    metadata[str(nodes)] = [c]
        return metadata

    @property
    def num_components(self):
        """return the number of components (R, L, C)"""
        return len(self._component_metadata)

    @property
    def component_list(self):
        """return the list of components"""
        return [x for x in self._component_metadata.keys()]


if __name__ == '__main__':
    file_name = './raw_netlist/RLC_low_pass.net'
    advance_circuit = AdvanceCircuit(file_name)
    tf = transfer_func(advance_circuit, [['0'], ['NC_01']], [['0'], ['N002']])

