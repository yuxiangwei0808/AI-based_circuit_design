import PySpice

from advance_graph import AdvanceCircuit
from lcapy import *
from utils import encode_circuit, differential_circuits, select_population, save_circuit, discard_topologies, simulate_circuit

import numpy as np
import random
from sympy import Piecewise, symbols
from tqdm import tqdm
import time
import os
import subprocess


COMPONENT_TYPES = ['L', 'C', 'R']
E12_SERIES = [1.0, 1.2, 1.5, 1.8, 2.2, 2.7, 3.3, 3.9, 4.7, 5.6, 6.8, 8.2]
POWER = [10**i for i in range(-9, 3)]
MAX_ITER = 100


def parameter_mutate(circuit):
    altered_components = random.sample(circuit.component_list, k=random.randint(1, circuit.num_components))
    for c in altered_components:
        component_value = random.choice(E12_SERIES) * random.choice(POWER)
        circuit.alter_component_value(c, component_value)

    return circuit


def parameter_worker(population_metadata, label_topo, pop_size_per_topo=50, per_topo_iter=30, recombination_rate=0.5):
    population_new, population_metadata_new = {}, {}
    per_topo_pop = population_metadata[label_topo]
    init_circuit = per_topo_pop[list(per_topo_pop.keys())[0]][0]
    org_size = len(population_metadata[label_topo])
    for _ in range(pop_size_per_topo - org_size):
        new_circuit = parameter_mutate(init_circuit)
        new_fitness = simulate_circuit(new_circuit)
        label_param = encode_circuit(new_circuit, True)
        per_topo_pop.update({label_param: [new_circuit, new_fitness]})

    for _ in range(per_topo_iter):
        per_topo_pop_new = per_topo_pop.copy()
        for label in per_topo_pop:
            cross_over = random.random()
            if cross_over <= recombination_rate:
                candidates = per_topo_pop.copy()
                target, fitness_target = per_topo_pop[label]
                # mutate
                candidates.pop(label)
                x_1, x_2, x_3 = random.sample(list(candidates.keys()), 3)
                x_1, x_2, x_3 = candidates[x_1][0], candidates[x_2][0], candidates[x_3][0]
                candidate = x_3.deepcopy()

                diff = differential_circuits(x_1, x_2, candidate)
                for k in diff:
                    candidate.alter_component_value(k, diff[k])

                fitness_candidate = simulate_circuit(candidate)

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
    population_metadata_new.update({label_topo: per_topo_pop_new})

    if not os.path.isdir('./checkpoint/param_opt_1/' + label_topo):
        os.mkdir('./checkpoint/param_opt_1/' + label_topo)
    path = './checkpoint/param_opt_1/' + label_topo
    save_circuit(population_new, path=path)

    return population_new, population_metadata_new


def parameter_opt(population, population_metadata):
    """perform parameter optimization for each topology"""
    population_new, population_metadata_new = {}, {}

    labels = list(population_metadata.keys())
    s = time.time()
    pop_per_topo = []
    for label in tqdm(labels, desc='parameter optimize'):
        t = parameter_worker(population_metadata, label)
        pop_per_topo.append(t)

    for p, p_m in pop_per_topo:
        population_new.update(p)
        population_metadata_new.update(p_m)
    print(f'totally consume:{time.time() - s} to optimize parametes of {len(labels)} topologies')

    return population_new, population_metadata_new


def main(path):
    """resume POM based on a evolved topology"""
    population = {}
    for netlist in
