"""
Evolve based on an RLC low-pass filter. Initially we set the fc=1kHz.
R1 N001 NC_01 1.2
L1 N002 N001 1e-4
C1 N002 0 2.2e-4
Vin is set to be voltage across node NC_01 and node 0 (ground),
Vout is set to be the voltage across node N002 and node 0.
"""
import shutil

import PySpice

from advance_graph import AdvanceCircuit
from lcapy import *
from utils import encode_circuit, get_lowest_fitness, select_population, save_circuit, discard_topologies

import numpy as np
import random
from sympy import Piecewise, symbols
from tqdm import tqdm
import time
import sys
import os
from collections import Counter
import contextlib


COMPONENT_TYPES = ['C', 'R', 'Q']
E12_SERIES = [1.0, 1.2, 1.5, 1.8, 2.2, 2.7, 3.3, 3.9, 4.7, 5.6, 6.8, 8.2]
POWER = {'R': [10**i for i in range(1, 6)], 'C': [10**i for i in range(-10, -5)]}
EXCLUDED_ELEMENTS = ['C2', 'R1', 'C1', 'C5']


def init_mutate(new_circuit):
    init_node_list = new_circuit.node_list
    init_branch_list = new_circuit.branch_list
    init_branch_list = [x for x in init_branch_list if x not in EXCLUDED_ELEMENTS]
    gen = random.choice([0, 1, 2])
    if gen == 0:  # connect two nodes
        component_type = random.choice(COMPONENT_TYPES)
        component_name = component_type + '_' + str(random.randint(0, 1e6))
        if component_type[0] == 'Q':  # add a transistor
            while True:
                nodes = random.sample(init_node_list, k=3)
                if not ('0' in nodes and 'NC_01' in nodes):
                    break
            new_circuit.connect_three_nodes(nodes, component_name)
        else:
            while True:
                node_pair = random.sample(init_node_list, k=2)
                if Counter(node_pair) != Counter(['0', 'NC_01']):
                    break
            component_value = random.choice(E12_SERIES) * random.choice(POWER[component_type])
            new_circuit.connect_two_nodes(node_pair, [component_name, component_value])
    elif gen == 1:  # insert a new node
        branch = random.choice(init_branch_list)  # the selected component to be relocated
        if branch[0] == 'Q':  # add components to the edge that connects a BJT
            nodes = list(new_circuit[branch].relnodes)
            node0 = random.choice(nodes)  # randomly choose a node of the BJT
            component_type = random.choice(COMPONENT_TYPES)
            component_name = component_type + '_' + str(random.randint(0, 1e6))
            if component_type == 'Q':
                node1 = random.choice([i for i in init_node_list if i != node0])
                new_circuit.insert_new_node_to_bjt([node0, node1], component_name, branch)
            else:
                component_value = random.choice(E12_SERIES) * random.choice(POWER[component_type])
                new_circuit.insert_new_node_to_bjt([node0], component_name, branch, component_value)
        else:
            component_type = random.choice(COMPONENT_TYPES)
            node_pair = list(new_circuit[branch].relnodes)
            random.shuffle(node_pair)
            component_name = component_type + '_' + str(random.randint(0, 1e6))
            if component_type == 'Q':  # add BJT
                node3 = random.choice([i for i in init_node_list if i not in node_pair])
                node_pair.append(node3)
                new_circuit.insert_new_node_bjt(node_pair, component_name)
            else:
                component_value = random.choice(E12_SERIES) * random.choice(POWER[component_type])
                new_circuit.insert_new_node(node_pair, [component_name, component_value])
    else:  # alter values only
        components = init_branch_list.copy()
        components = [c for c in components if c[0] != 'Q']
        altered_components = random.choice(components)
        component_type = altered_components[0]
        component_value = random.choice(E12_SERIES) * random.choice(POWER[component_type])
        new_circuit.alter_component_value(altered_components, component_value)

    return new_circuit


def topology_mutate(circuit_org, max_num_components=28):
    """mutation with three conditions: connect two nodes, insert a new node, change node connection"""
    circuit = circuit_org.deepcopy()
    node_list = circuit.node_list
    branch_list = circuit.branch_list
    branch_list = [x for x in branch_list if x not in EXCLUDED_ELEMENTS]

    if circuit.num_components <= max_num_components:  # mutation involves element change
        gen = random.choice([0, 1, 2, 3])
        if gen == 0:  # connect two nodes
            component_type = random.choice(COMPONENT_TYPES)
            component_name = component_type + '_' + str(random.randint(0, 1e6))
            if component_type[0] == 'Q':  # add a transistor
                while True:
                    nodes = random.sample(node_list, k=3)
                    if not ('0' in nodes and 'NC_01' in nodes):
                        break
                circuit.connect_three_nodes(nodes, component_name)
            else:
                while True:
                    node_pair = random.sample(node_list, k=2)
                    if Counter(node_pair) != Counter(['0', 'NC_01']):
                        break
                component_value = random.choice(E12_SERIES) * random.choice(POWER[component_type])
                circuit.connect_two_nodes(node_pair, [component_name, component_value])
        elif gen == 1:  # insert a new node
            branch = random.choice(branch_list)  # the selected component to be relocated
            if branch[0] == 'Q':  # add components to the edge that connects a BJT
                nodes = list(circuit[branch].relnodes)
                node0 = random.choice(nodes)  # randomly chosse a node of the BJT
                component_type = random.choice(COMPONENT_TYPES)
                component_name = component_type + '_' + str(random.randint(0, 1e6))
                if component_type == 'Q':
                    node1 = random.choice([i for i in node_list if i != node0])
                    circuit.insert_new_node_to_bjt([node0, node1], component_name, branch)
                else:
                    component_value = random.choice(E12_SERIES) * random.choice(POWER[component_type])
                    circuit.insert_new_node_to_bjt([node0], component_name, branch, component_value)
            else:
                component_type = random.choice(COMPONENT_TYPES)
                node_pair = list(circuit[branch].relnodes)
                random.shuffle(node_pair)
                component_name = component_type + '_' + str(random.randint(0, 1e6))
                if component_type == 'Q':  # add BJT
                    node3 = random.choice([i for i in node_list if i not in node_pair])
                    node_pair.append(node3)
                    circuit.insert_new_node_bjt(node_pair, component_name)
                else:
                    component_value = random.choice(E12_SERIES) * random.choice(POWER[component_type])
                    circuit.insert_new_node(node_pair, [component_name, component_value])
        elif gen == 2:  # delete an element
            branch = random.choice(branch_list)
            circuit.delete_component(branch)
        elif gen == 3:  # change connection
            branch = random.choice(branch_list)
            if branch[0] == 'Q':
                while True:
                    nodes = random.sample(node_list, k=3)
                    if not ('0' in nodes and 'NC_01' in nodes):
                        break
            else:
                while True:
                    nodes = random.sample(node_list, k=2)
                    if Counter(nodes) != Counter(['0', 'NC_01']):
                        break
            circuit.change_element_connection(branch, nodes)
    else:  # mutation only involves topology change
        branch = random.choice(branch_list)
        gen = random.choice([0, 1])
        if gen == 0:
            if branch[0] == 'Q':
                while True:
                    nodes = random.sample(node_list, k=3)
                    if not ('0' in nodes and 'NC_01' in nodes):
                        break
            else:
                while True:
                    nodes = random.sample(node_list, k=2)
                    if Counter(nodes) != Counter(['0', 'NC_01']):
                        break
            circuit.change_element_connection(branch, nodes)
        elif gen == 1:
            branch = random.choice(branch_list)
            circuit.delete_component(branch)

    return circuit


def topology_opt(population, population_metadata, pop_size, max_iter=100, recombination_rate=0.5):
    # check if the number of topologies is equal to the population size
    if len(population_metadata) < pop_size:
        dif = pop_size - len(population_metadata)
        i = 0
        while True:
            candidate_labels = list(population.keys())
            candidate_label = random.choice(candidate_labels)
            candidate = population[candidate_label][0]
            new_circuit = topology_mutate(candidate)
            label_param, label_topo = encode_circuit(new_circuit, True), encode_circuit(new_circuit, False)
            try:
                with contextlib.redirect_stderr(None):
                    fitness = simulate_amp(new_circuit)
                population.update({label_param: [new_circuit, fitness]})
                if label_topo in population_metadata:
                    population_metadata[label_topo].update({label_param: [new_circuit, fitness]})
                else:
                    population_metadata.update({label_topo: {label_param: [new_circuit, fitness]}})
                i += 1
            except PySpice.Spice.NgSpice.Shared.NgSpiceCommandError:
                continue
            if not i < dif:
                break

    print('begin optimize topology')
    # cycle through each topology in the population
    for _ in tqdm(range(max_iter), desc='topology optimize', file=sys.stdout):
        population_copy = population.copy()
        for k, target_label_param in enumerate(population_copy):
            target, fitness_target = tuple(population_copy[target_label_param])
            target_label_topo = encode_circuit(target, False)

            crossover = random.random()
            if crossover <= recombination_rate:
                candidate_labels = list(population.keys())
                candidate_labels.remove(target_label_param)
                candidate_label = random.choice(candidate_labels)
                candidate = population[candidate_label][0]
                cnt = 0
                while True:
                    # mutate
                    mutated_candidate = topology_mutate(candidate)
                    mutated_label_topo = encode_circuit(mutated_candidate, with_parameter=False)
                    try:
                        with contextlib.redirect_stderr(None):
                            fitness_mutated = simulate_amp(mutated_candidate)
                        if fitness_mutated < fitness_target:  # substitute
                            mutate_label_parameter = encode_circuit(mutated_candidate, True)

                            population.pop(target_label_param)
                            population.update({mutate_label_parameter: [mutated_candidate, fitness_mutated]})

                            population_metadata[target_label_topo].pop(target_label_param)
                            if mutated_label_topo in population_metadata:
                                population_metadata[mutated_label_topo].update({mutate_label_parameter: [mutated_candidate, fitness_mutated]})
                            else:
                                population_metadata.update({mutated_label_topo: {mutate_label_parameter: [mutated_candidate, fitness_mutated]}})
                        else:  # no change to the population
                            pass
                        break
                    except PySpice.Spice.NgSpice.Shared.NgSpiceCommandError:  # repeat due to the invalid circuit
                        if cnt < 10:  # set a maximum number of retries
                            cnt += 1
                            continue
                        else:
                            break
        get_lowest_fitness(population)
        # os.system('rm -rf ./checkpoint/amplifier/topo_opt_3/*')
        shutil.rmtree('./checkpoint/amplifier/topo_opt_3')
        save_circuit(population, path='./checkpoint/amplifier/topo_opt_3')
    # iter through topologies and delete topology that has no corresponding circuit
    population_metadata_copy = population_metadata.copy()
    for label_topo in population_metadata:
        if population_metadata[label_topo] == {}:
            population_metadata_copy.pop(label_topo)

    return population, population_metadata_copy


def optimization(init_circuit, init_pop_size, checkpoint=None):
    # generate initial population
    init_population = {}
    init_population_metadata = {}  # {label_topology: {label_parameter: [circuit, fitness]})} store circuit's indices with the same topology.

    if checkpoint is None:
        print('generate initial population')
        s = time.time()
        i = 0
        fitness = simulate_amp(init_circuit)
        label_param, label_topo = encode_circuit(init_circuit, True), encode_circuit(init_circuit, False)
        init_population.update({label_param: [init_circuit, fitness]})
        init_population_metadata.update({label_topo: {label_param: [init_circuit, fitness]}})
        while True:
            new_circuit = init_circuit.deepcopy()
            step = random.randint(1, 3)
            for _ in range(step):
                new_circuit = init_mutate(new_circuit)
            label_topo = encode_circuit(new_circuit, with_parameter=False)
            label_parameter = encode_circuit(new_circuit, with_parameter=True)
            try:
                with contextlib.redirect_stderr(None):
                    fitness = simulate_amp(new_circuit)
                init_population.update({label_parameter: [new_circuit, fitness]})
                if label_topo in init_population_metadata:
                    init_population_metadata[label_topo].update({label_parameter: [new_circuit, fitness]})
                else:
                    init_population_metadata.update({label_topo: {label_parameter: [new_circuit, fitness]}})
                i += 1
            except PySpice.Spice.NgSpice.Shared.NgSpiceCommandError:
                continue
            if not i < init_pop_size:
                break
        print('finished generating population')
        print(f'consume {time.time() - s}s')

        population, population_metadata = topology_opt(init_population, init_population_metadata, init_pop_size,
                                                       max_iter=500)
        population, population_metadata = discard_topologies(population, population_metadata, 200)
        print(len(population))
        save_circuit(population_metadata, path='./checkpoint/amplifier/topo_opt_1', accord_topo=True)
    else:
        population, population_metadata = {}, {}
        files = os.listdir(checkpoint)
        for file in files:
            file_path = os.path.join(checkpoint, file)
            circuit = AdvanceCircuit(file_path)
            label_param, label_topo = encode_circuit(circuit, True), encode_circuit(circuit, False)
            fitness = simulate_amp(circuit)

            population.update({label_param: [circuit, fitness]})
            assert label_topo not in population_metadata
            population_metadata.update({label_topo: {label_param: [circuit, fitness]}})

        population, population_metadata = topology_opt(population, population_metadata, init_pop_size, max_iter=420)
        # population, population_metadata = discard_topologies(population, population_metadata, 100)
        print(len(population))
        # os.system('rm -rf ./checkpoint/amplifier/topo_opt_3/*')
        shutil.rmtree('./checkpoint/amplifier/topo_opt_3')
        save_circuit(population_metadata, path='./checkpoint/amplifier/topo_opt_3', accord_topo=True)



if __name__ == '__main__':
    file_name = './raw_netlist/amplifier.net'
    init_circuit = AdvanceCircuit(file_name)

    optimization(init_circuit, 300, checkpoint='./checkpoint/amplifier/topo_opt_3')
