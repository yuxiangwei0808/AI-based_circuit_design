import random
import json
import hashlib


def connect_two_node(circuit, node_pair: list, component_metadata: list):
    """
    TODO: support three-terminal component
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


def change_element_connection(circuit, element, new_pair, value):
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
    ist = [element, new_pair[0], new_pair[1], str(value)]
    ist = ' '.join(ist)
    circuit.add(ist)

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


def select_population(population: dict, K=2):
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
    differential = {k: components_c1[k] - components_c2[k] + mutate_factor * components_c3[k] for k in components_c1}
    for k in differential:
        if differential[k] <= 0:
            differential[k] = components_c3[k]

    return differential
