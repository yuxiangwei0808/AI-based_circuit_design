"""
Evolve based on an RLC low-pass filter. Initially we set the fc=1kHz.
R1 N001 NC_01 1.2
L1 N002 N001 1e-4
C1 N002 0 2.2e-4
Vin is set to be voltage across node NC_01 and node 0 (ground),
Vout is set to be the voltage across node N002 and node 0.
"""

from advance_graph import AdvanceCircuit

from lcapy import *
from utils import transfer_func, encode_circuit, differential_circuits, select_population, save_circuit
import numpy as np
import random
from sympy import Piecewise, symbols
from tqdm import tqdm
import time
import multiprocessing as mp
import pickle
import dill
import joblib
import traceback


COMPONENT_TYPES = ['L', 'C', 'R']
E12_SERIES = [1.0, 1.2, 1.5, 1.8, 2.2, 2.7, 3.3, 3.9, 4.7, 5.6, 6.8, 8.2]
POWER = [10**i for i in range(-9, 3)]
MAX_ITER = 100


def init_mutate(new_circuit):
    init_node_list = new_circuit.node_list
    init_branch_list = new_circuit.branch_list
    gen = random.choice([0, 1, 2])
    if gen == 0:  # connect two nodes
        node_pair = random.sample(init_node_list, k=2)
        component_name = random.choice(COMPONENT_TYPES) + '0_' + str(random.randint(0, 1e6))
        component_value = random.choice(E12_SERIES) * random.choice(POWER)
        new_circuit.connect_two_nodes(node_pair, [component_name, component_value])
    elif gen == 1:  # insert a new node
        branch = random.choice(init_branch_list)
        node_pair = list(new_circuit[branch].relnodes)
        random.shuffle(node_pair)
        component_name = random.choice(COMPONENT_TYPES) + '0_' + str(random.randint(0, 1e6))
        component_value = random.choice(E12_SERIES) * random.choice(POWER)
        new_circuit.insert_new_node(node_pair, [component_name, component_value])
    else:  # alter values only
        altered_components = random.sample(new_circuit.component_list, k=random.randint(1, new_circuit.num_components))
        for c in altered_components:
            component_value = random.choice(E12_SERIES) * random.choice(POWER)
            new_circuit.alter_component_value(c, component_value)

    return new_circuit


def topology_mutate(circuit_org, topo_mut_threshold=0.5, max_num_components=28):
    """mutation with three conditions: connect two nodes, insert a new node, change node connection"""
    circuit = circuit_org.deepcopy()
    node_list = circuit.node_list
    branch_list = circuit.branch_list

    if random.random() > topo_mut_threshold and circuit.num_components <= max_num_components:  # mutation involves element change
        gen = random.choice([0, 1])
        if gen == 0:  # connect two nodes
            node_pair = random.sample(node_list, k=2)
            component_name = random.choice(COMPONENT_TYPES) + '0_' + str(random.randint(0, 1e6))
            component_value = random.choice(E12_SERIES) * random.choice(POWER)
            circuit.connect_two_nodes(node_pair, [component_name, component_value])
        elif gen == 1:  # insert a new node
            branch = random.choice(branch_list)
            node_pair = list(circuit[branch].relnodes)
            random.shuffle(node_pair)
            component_name = random.choice(COMPONENT_TYPES) + '0_' + str(random.randint(0, 1e6))
            component_value = random.choice(E12_SERIES) * random.choice(POWER)
            circuit.insert_new_node(node_pair, [component_name, component_value])
    else:  #  mutation only involves topology change
        branch = random.choice(branch_list)
        new_node_pair = random.sample(node_list, k=2)
        circuit.change_element_connection(branch, new_node_pair)

    return circuit


def parameter_mutate(circuit):
    altered_components = random.sample(circuit.component_list, k=random.randint(1, circuit.num_components))
    for c in altered_components:
        component_value = random.choice(E12_SERIES) * random.choice(POWER)
        circuit.alter_component_value(c, component_value)

    return circuit


def fitness_func(tf):
    """calculate the fitness based on the transfer function"""
    x = symbols('x')
    idea_LPF = Piecewise((1, x < 1e3*2*pi), (0, True))  # ideal fileter with cutoff frequency 1kHZ

    sample_points = list(np.arange(100, 1e4+100, 100)*2*pi)
    freq_response = tf(jw)
    gain = abs(freq_response)

    result = [abs(idea_LPF.subs(x, point) - gain(point).evalf()) for point in sample_points]
    fit = sum(result)
    fit = fit.evalf()

    return fit


def topology_opt(population, population_metadata, max_iter=100, recombination_rate=0.5):
    # cycle through each topology in the population
    for _ in tqdm(range(max_iter), desc='topology optimize'):
        population_copy = population.copy()
        for k, target_label_param in enumerate(population_copy):
            target, fitness_target = tuple(population_copy[target_label_param])
            target_label_topo = encode_circuit(target, False)

            crossover = random.random()
            if crossover <= recombination_rate:
                candidate_labels = list(population.keys())
                candidate_label = random.choice(candidate_labels)
                candidate = population[candidate_label][0]

                # mutate
                mutated_candidate = topology_mutate(candidate)
                mutated_label_topo = encode_circuit(mutated_candidate, with_parameter=False)

                if mutated_label_topo in population_metadata:
                    flag = 0
                    Vin_tree, Vout_tree, _ = population_metadata[mutated_label_topo]
                else:
                    flag = 1
                    exclude_pair_in = [['0'], ['NC_01']]  # [['0'], ['NC_01']]
                    Vin_tree = mutated_candidate.find_two_tree(exclude_pair_in)
                    Vout_nodes = [['0'], ['N002']]
                    exclude_pair_out = [Vout_nodes[0], Vout_nodes[1] + ['NC_01']]
                    Vout_tree = mutated_candidate.find_two_tree(exclude_pair_out)
                if Vin_tree and Vout_tree:
                    mutated_tf = transfer_func(mutated_candidate, Vin_tree, Vout_tree)
                    fitness_mutated = fitness_func(mutated_tf)

                    if fitness_mutated < fitness_target:  # substitute
                        population.pop(target_label_param)
                        mutate_label_parameter = encode_circuit(mutated_candidate, True)
                        population.update({mutate_label_parameter: [mutated_candidate, fitness_mutated]})
                        population_metadata[target_label_topo][-1].pop(target_label_param)
                        if flag == 0:
                            population_metadata[mutated_label_topo][-1][mutate_label_parameter] = [mutated_candidate, fitness_mutated]
                        else:
                            population_metadata[mutated_label_topo] = \
                                (Vin_tree, Vout_tree, {mutate_label_parameter: [mutated_candidate, fitness_mutated]})
            else:  # no change to the population
                pass

    # iter through topologies and delete topology that has no corresponding circuit
    population_metadata_copy = population_metadata.copy()
    for label_topo in population_metadata:
        if population_metadata[label_topo][-1] == {}:
            population_metadata_copy.pop(label_topo)

    return population, population_metadata_copy


def parameter_worker(population_metadata, label_topo, pop_size_per_topo=20, per_topo_iter=10, recombination_rate=0.5):
    population_new, population_metadata_new = {}, {}
    Vin_tree, Vout_tree, per_topo_pop = population_metadata[label_topo]
    init_circuit = per_topo_pop[list(per_topo_pop.keys())[0]][0]
    org_size = len(population_metadata[label_topo][-1])
    print(f'begin generate population for the topology {label_topo}')
    s = time.time()
    for _ in range(pop_size_per_topo - org_size):
        new_circuit = parameter_mutate(init_circuit)
        new_tf = transfer_func(new_circuit, Vin_tree, Vout_tree)
        new_fitness = fitness_func(new_tf)
        label_param = encode_circuit(new_circuit, True)
        per_topo_pop.update({label_param: [new_circuit, new_fitness]})

    print(f'consume {time.time() - s}, begin optimize parameters')

    for _ in range(per_topo_iter):
        candidates = per_topo_pop.copy()
        per_topo_pop_new = per_topo_pop.copy()
        for label in per_topo_pop:
            target, fitness_target = per_topo_pop[label]
            cross_over = random.random()
            if cross_over <= recombination_rate:
                # mutate
                candidates.pop(label)
                x_1, x_2, x_3 = random.sample(list(candidates.keys()), 3)
                x_1, x_2, x_3 = candidates[x_1][0], candidates[x_2][0], candidates[x_3][0]
                candidate = x_3.deepcopy()

                diff = differential_circuits(x_1, x_2, candidate)
                for k in diff:
                    candidate.alter_component_value(k, diff[k])

                tf_candidate = transfer_func(candidate, Vin_tree, Vout_tree)
                fitness_candidate = fitness_func(tf_candidate)

                if fitness_candidate < fitness_target:
                    label_param_candidate = encode_circuit(candidate, True)
                    per_topo_pop_new.pop(label)
                    per_topo_pop_new.update({label_param_candidate: [candidate, fitness_candidate]})
                    if fitness_candidate < 1e-3:
                        break
                else:
                    pass

    # reduce the number of circuits in each topology. Default to top 1
    per_topo_pop_new = select_population(per_topo_pop_new)
    population_new.update(per_topo_pop_new)
    population_metadata_new.update({label_topo: (Vin_tree, Vout_tree, per_topo_pop_new)})

    return population_new, population_metadata_new


def mp_worker(args):
    population_metadata, label = args[0], args[1]
    p, p_m = parameter_worker(population_metadata, label)
    return p, p_m


def parameter_opt(population, population_metadata, pop_size_per_topo=20, per_topo_iter=10, recombination_rate=0.5):
    """perform parameter optimization for each topology"""
    population_new, population_metadata_new = {}, {}

    labels = list(population_metadata.keys())
    args = [(population_metadata, label) for label in labels]
    s = time.time()
    with mp.pool.ThreadPool(mp.cpu_count()) as pool:
        pop_per_topo = pool.map(mp_worker, args)

    for p, p_m in pop_per_topo:
        population_new.update(p)
        population_metadata_new.update(p_m)
    print(f'totally consume:{time.time() - s}')

    return population_new, population_metadata_new


def optimization(init_circuit, init_pop_size):
    # generate initial population
    init_population = {}
    init_population_metadata = {}  # {label_topology: (in, out, {label_parameter: [circuit, fitness]})} store circuit's indices with the same topology.
    for i in tqdm(range(init_pop_size), desc='generate initial population'):
        new_circuit = init_circuit.deepcopy()
        step = random.randint(1, 3)
        for _ in range(step):
            new_circuit = init_mutate(new_circuit)

        label = encode_circuit(new_circuit, with_parameter=False)
        label_parameter = encode_circuit(new_circuit, with_parameter=True)
        if label in init_population_metadata:
            Vin_tree, Vout_tree, _ = init_population_metadata[label]
            init_population_metadata[label][-1].update({label_parameter: [new_circuit]})
        else:
            exclude_pair_in = [['0'], ['NC_01']]  # [['0'], ['NC_01']]
            Vin_tree = new_circuit.find_two_tree(exclude_pair_in)
            Vout_nodes = [['0'], ['N002']]
            exclude_pair_out = [Vout_nodes[0], Vout_nodes[1] + ['NC_01']]
            Vout_tree = new_circuit.find_two_tree(exclude_pair_out)
            init_population_metadata[label] = (Vin_tree, Vout_tree, {label_parameter: [new_circuit]})

        tf = transfer_func(new_circuit, Vin_tree, Vout_tree)
        fitness = fitness_func(tf)  # TODO: optimize this as this is quite time-consuming
        init_population_metadata[label][-1][label_parameter].append(fitness)
        init_population.update({label_parameter: [new_circuit, fitness]})

    population, population_metadata = topology_opt(init_population, init_population_metadata, max_iter=30)
    population, population_metadata = parameter_opt(population, population_metadata)
    save_circuit(population)

    population, population_metadata = topology_opt(population, population_metadata, max_iter=30)
    population, population_metadata = parameter_opt(population, population_metadata)
    save_circuit(population)

    population, population_metadata = topology_opt(population, population_metadata, max_iter=30)
    population, population_metadata = parameter_opt(population, population_metadata)
    save_circuit(population)


if __name__ == '__main__':
    file_name = './raw_netlist/RLC_low_pass.net'
    init_circuit = AdvanceCircuit(file_name)

    optimization(init_circuit, 50)
