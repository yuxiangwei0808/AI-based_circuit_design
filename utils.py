from lcapy import Circuit

_NEW_NODE = 0


def connect_two_node(circuit: Circuit, node_pair: tuple, component_metadata: tuple):
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


def insert_new_node(circuit: Circuit, node_pair: tuple, component_metadata: tuple):
    """
    Choose a node1 and an edge (another node2 connect to the node1) and insert a new node3 on the edge.
    The original component is between node1 and 2, and the added component id between node2 and 3.
    Args:
        circuit: Circuit object
        node_pair: (node1, node2)
        component_metadata: a length-2 list with the form [component_name, value]

    Returns:
        new Circuit object or 0 if no edge found for the specified node pair
    """
    global _NEW_NODE
    component_nodes, component_name, component = circuit._find_component_by_nodes(node_pair)
    node1, node2 = node_pair[0], node_pair[1]
    node3 = 'S' + str(_NEW_NODE)
    component_value = get_value(component)

    circuit.remove(component_name)

    instruction1 = [component_name, node1, node3, str(component_value)]
    instruction1 = ' '.join(instruction1)
    circuit.add(instruction1)

    instruction2 = [str(component_metadata[0]), node3, node2, str(component_metadata[1])]
    instruction2 = ' '.join(instruction2)
    circuit.add(instruction2)

    _NEW_NODE += 1
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
