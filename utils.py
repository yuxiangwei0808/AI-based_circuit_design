import random
import json
import hashlib
import time

import numpy as np
from sympy import Piecewise, symbols
import os
import collections
import subprocess

import PySpice
from PySpice.Spice.Parser import SpiceParser
from PySpice.Unit import *


def connect_two_node(circuit, node_pair: list, component_metadata: list):
    """
    connect two existing nodes on the Circuit object with a new edge;
    Args:
        circuit: Circuit object
        node_pair: nodes to be connected with a new component
        component_metadata: a length-2 list with the form [component_name, value]

    Returns:
        new Circuit object
    """
    instruction = [str(component_metadata[0]), str(node_pair[0]), str(node_pair[1]), str(component_metadata[1])]
    instruction = ' '.join(instruction)
    circuit.add(instruction)
    return circuit


def connect_three_nodes(circuit, nodes: list, name: str):
    """
    connect three nodes by a three-terminal component, such as BJT
    Args:
        circuit: Circuit
        nodes: base, emitter, collecter
        name: BJT name

    Returns:
        new Circuit
    """
    ist = [str(x) for x in nodes]
    ist.insert(0, name)
    ist.append('2N3904')  # add BJT 2N3904
    ist = ' '.join(ist)
    circuit.add(ist)
    return circuit


def insert_new_node(circuit, node_pair: list, component_metadata: list):
    """
    Choose a node1 and an edge (another node2 connect to node1) and insert a new node3 on the edge.
    The original component is between node1 and 3, and the added component id between node2 and 3.
    Args:
        circuit: Circuit object
        node_pair: (node1, node2)
        component_metadata: a length-2 list with the form [component_name, value]

    Returns:
        new Circuit object or Exception if no edge found for the specified node pair
    """
    component_nodes, component_name, component = circuit._find_component_by_nodes(node_pair)
    node1, node2 = node_pair[0], node_pair[1]
    node3 = 'S' + str(random.randint(0, 1e6))
    component_value = get_value(component)

    circuit.remove(component_name)

    instruction1 = [component_name, node1, node3, str(component_value)]
    instruction1 = ' '.join(instruction1)
    circuit.add(instruction1)

    instruction2 = [str(component_metadata[0]), node3, node2, str(component_metadata[1])]
    instruction2 = ' '.join(instruction2)
    circuit.add(instruction2)

    return circuit


def insert_new_node_bjt(circuit, nodes: list, name: str):
    """
    insert a new node then connect it with a BJT.
    a new node4 is added between node1 and 2; the original component is put between node1 and 4.
    (npn BJT) Collector--node4, base--node2, emitter--node3.
    Args:
        circuit: Circuit
        nodes: [node1, node2, node3]
        name: BJT name

    Returns:
        Circuit
    """
    component_nodes, component_name, component = circuit._find_component_by_nodes(nodes[:2])
    node4 = 'S' + str(random.randint(0, 1e6))
    component_value = get_value(component)

    ist1 = [component_name, nodes[0], node4, str(component_value)]
    ist1 = ' '.join(ist1)
    circuit.add(ist1)

    ist2 = [name, node4, nodes[1], nodes[2], '2N3904']
    ist2 = ' '.join(ist2)
    circuit.add(ist2)

    return circuit


def insert_new_node_to_bjt(circuit, nodes: list, name: str, bjt_name: str, new_component_value: str = None):
    """
    insert a node (node2) to the edge that connect to a BJT node (node0). Consider add a BJT or other components.
    For two-terminal components, the added one is between node0 and node2, the BJT node is now node2.
    For BJT, the added collector--node2, base--node0, emitter--node1
    Args:
        circuit: Circuit
        nodes: [node0] or [node0, node1] if add a BJT
        name: added component name
        bjt_name: BJT name
        new_component_value: value of the added C, L, or R

    Returns:
        Circuit
    """
    node2 = 'S' + str(random.randint(0, 1e6))
    bjt_nodes = list(circuit.elements[bjt_name].relnodes)
    bjt_nodes[bjt_nodes.index(nodes[0])] = node2
    ist1 = bjt_nodes
    ist1.insert(0, bjt_name)
    ist1.append('2N3904')
    ist1 = ' '.join(ist1)
    circuit.add(ist1)

    if name[0] == 'Q':
        assert len(nodes) == 2
        ist2 = [name, node2, nodes[0], nodes[1], '2N3904']
    else:
        assert len(nodes) == 1 and new_component_value is not None
        ist2 = [name, nodes[0], node2, str(new_component_value)]
    ist2 = ' '.join(ist2)
    circuit.add(ist2)

    return circuit


def change_element_connection(circuit, element, new_pair, value=None):
    """
    change an element's connection from old_pair to new_pair.
    Args:
        circuit: Circuit object
        element: element name
        new_pair: tuple
        value: element value

    Returns:
        new Circuit
    """
    ist = [element]
    if element[0] == 'Q':
        ist.extend(new_pair)
        ist.append('2N3904')
    else:
        new_pair.append(str(value))
        ist.extend(new_pair)

    ist = ' '.join(ist)
    circuit.add(ist)

    return circuit


def connect_standalone_node_to_ground(circuit, in_out_nodes: list):
    all_nodes = [circuit._component_metadata[x][0] for x in circuit._component_metadata]
    all_nodes = [y for x in all_nodes for y in x]
    counter = dict(collections.Counter(all_nodes))
    for node in counter.keys():
        if node != '0' and node not in in_out_nodes and counter[node] < 2:
            for c in circuit._component_metadata:
                if node in circuit._component_metadata[c][0]:
                    node_pair = list(circuit._component_metadata[c][0])
                    node_pair.remove(node)
                    node_pair.append('0')
                    circuit = change_element_connection(circuit, c, node_pair, circuit._component_metadata[c][-1])
    return circuit


def get_value(component):
    name, c_type = component.name, component.type
    if c_type == 'R':
        value = component.R
    elif c_type == 'L':
        value = component.L
    elif c_type == 'C':
        value = component.C
    elif c_type == 'V':
        value = component.V
    else:
        raise NotImplementedError
    return value


def nodes_product(pools: list):
    """
    input should be: [[pair1], [pair2]]
    For example: [[a, b], [c, d]] -> ac, ad, bc, bd
    """
    result = [[]]
    for pool in pools:
        if not isinstance(pool, list):
            pool = [pool]
        result = [x + [y] for x in result for y in pool]
    result = [tuple(r) for r in result]
    return result


def transfer_func(circuit, Vin_tree, Vout_tree):
    """currently only considers the RLC low-pass filter circuit"""
    product_in = circuit.compute_tree_product(Vin_tree)
    product_out = circuit.compute_tree_product(Vout_tree)

    return product_out / product_in


def encode_circuit(circuit, with_parameter: bool):
    """give circuits a unique label"""
    e = dict(circuit.elements)
    if with_parameter:
        for i in e:
            e[i] = str(e[i])
    else:
        for i in e:
            ele_meta = str(e[i])
            pos_value = ele_meta.rfind(' ')
            e[i] = ele_meta[:pos_value]

    string = json.dumps(e, sort_keys=True).encode()
    label = hashlib.md5(string).hexdigest()

    return label


def select_population(population: dict, K=1):
    """select top K circutis from each topology"""
    circuit_fitness_dict = population
    if len(circuit_fitness_dict) > K:
        sorted_fitness = dict(sorted(circuit_fitness_dict.items(), key=lambda item: item[-1][-1]))
        sorted_fitness_copy = sorted_fitness.copy()
        for i, label in enumerate(sorted_fitness_copy):
            if i < K:
                pass
            else:
                sorted_fitness.pop(label)
        population = sorted_fitness
    else:
        pass

    return population


def differential_circuits(c1, c2, c3, mutate_factor=1.):
    """calculate difference between two circuits with the same topology"""
    components_c1 = {k: c1._component_metadata[k][-1] for k in c1._component_metadata}
    components_c2 = {k: c2._component_metadata[k][-1] for k in c2._component_metadata}
    components_c3 = {k: c3._component_metadata[k][-1] for k in c3._component_metadata}
    altered_components = list(components_c1.keys())
    altered_components = [x for x in altered_components if x[0] != 'Q']
    differential = {k: components_c1[k] - components_c2[k] + mutate_factor * components_c3[k] for k in
                    altered_components}
    for k in differential:
        if differential[k] < 0:
            differential[k] = abs(differential[k])
        elif differential[k] == 0:
            differential[k] = components_c3[k]

    return differential


def save_circuit(population, path='./checkpoint', top=500, accord_topo=False):
    """save the circuits as checkpoint. TODO: fix pickle issue. Currently only save netlist files"""
    if not os.path.isdir(path):
        os.mkdir(path)

    if accord_topo:  # save the circuit of each topology
        for label_topo in population:
            per_topo_pop = population[label_topo]
            for label_param in per_topo_pop:
                circuit, fitness = tuple(per_topo_pop[label_param])
                fitness = str(fitness)[:5].replace('.', '_')
                path_topo = path + '/' + str(label_topo)
                if not os.path.isdir(path_topo):
                    os.mkdir(path_topo)
                with open(path_topo + f'/{label_topo}-{fitness}.net', 'w') as f:
                    f.writelines(str(circuit))
    else:
        sorted_population = dict(sorted(population.items(), key=lambda item: item[-1][-1]))
        labels = list(sorted_population.keys())
        # print(f'minimum fitness: {sorted_population[labels[0]][-1]}')

        top = len(population) if top > len(population) else top
        for i in range(top):
            circuit, fitness = tuple(sorted_population[labels[i]])
            label_topo = encode_circuit(circuit, False)
            # print(f'circuit of fitness {fitness} saved')
            fitness = str(fitness)[:5].replace('.', '_')
            with open(path + f'/{label_topo}-{fitness}.net', 'w') as f:
                f.writelines(str(circuit))


def save_single_circuit(circuit, fitness, path):
    if not os.path.isdir(path):
        os.mkdir(path)
    label_topo = encode_circuit(circuit, False)
    fitness = str(fitness)[:5].replace('.', '_')
    with open(path + f'/{label_topo}-{fitness}.net', 'w') as f:
        f.writelines(str(circuit))


def discard_topologies(population, population_metadata, K=10):
    """discard the worst K circuit topologies after parameter optimizing"""
    sorted_population = dict(sorted(population.items(), key=lambda item: item[-1][-1]))
    reduced_population = {x: sorted_population[x] for i, x in enumerate(sorted_population.keys()) if
                          i < len(sorted_population) - K}
    reduced_population_metadata = {}

    for label_param in reduced_population:
        label_topo = encode_circuit(reduced_population[label_param][0], False)
        if label_topo not in reduced_population_metadata:
            reduced_population_metadata.update({label_topo: {label_param: reduced_population[label_param]}})
        else:
            reduced_population_metadata[label_topo].update({label_param: reduced_population[label_param]})

    return reduced_population, reduced_population_metadata


def get_lowest_fitness(population):
    sorted_population = dict(sorted(population.items(), key=lambda item: item[-1][-1]))
    fitness = [x[-1] for x in sorted_population.values()]
    print(fitness[0])




def sync_files_from_server():
    while True:
        subprocess.call(['scp', '-r', 'root@180.76.58.164:/home/checkpoint/amplifier/topo_opt_5/*', './checkpoint/amplifier/tmp/'])
        time.sleep(300)


if __name__ == '__main__':
    from advance_graph import AdvanceCircuit
    circuit = AdvanceCircuit('./test.txt')
