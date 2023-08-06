import contextlib

import PySpice

from advance_graph import AdvanceCircuit
from lcapy import *
from utils import encode_circuit, differential_circuits, select_population, save_circuit

import random
import time
import os
import argparse
from tqdm import tqdm
import signal
import psutil


COMPONENT_TYPES = ['C', 'R']
E12_SERIES = [1.0, 1.2, 1.5, 1.8, 2.2, 2.7, 3.3, 3.9, 4.7, 5.6, 6.8, 8.2]
POWER = {'R': [10**i for i in range(1, 6)], 'C': [10**i for i in range(-10, -5)]}
EXCLUDED_ELEMENTS = ['C2', 'R1', 'C1', 'C5']


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='./checkpoint/amplifier/topo_opt_2/38dbd672d0ce4633078265a49f9a84e9')
    parser.add_argument('--bar', type=int, default=0)
    return parser.parse_args()


def parameter_mutate(circuit):
    component_list = circuit.component_list
    component_list = [x for x in component_list if x[0] != 'Q']
    altered_components = random.sample(component_list, k=random.randint(1, circuit.num_components))
    for c in altered_components:
        component_value = random.choice(E12_SERIES) * random.choice(POWER[c[0]])
        circuit.alter_component_value(c, component_value)
    del component_list
    return circuit


def parameter_worker(per_topo_pop, label_topo, bar, pop_size_per_topo=300, per_topo_iter=300, recombination_rate=0.5):
    init_circuit = per_topo_pop[list(per_topo_pop.keys())[0]][0]
    org_size = len(per_topo_pop)
    cnt = 0
    while True:
        # if psutil.Process().memory_info().rss / (1024 * 1024 * 1024) > 10.:
        #     os.kill(os.getpid(), signal.SIGKILL)
        try:
            new_circuit = parameter_mutate(init_circuit)
            with contextlib.redirect_stderr(None):
                new_fitness = simulate_amp(new_circuit)
            label_param = encode_circuit(new_circuit, True)
            per_topo_pop.update({label_param: [new_circuit, new_fitness]})
            cnt += 1
        except PySpice.Spice.NgSpice.Shared.NgSpiceCommandError:
            pass
        if cnt >= pop_size_per_topo - org_size:
            break

    for _ in tqdm(range(per_topo_iter), desc=f'optimize {label_topo}', position=bar):
        # if psutil.Process().memory_info().rss / (1024 * 1024 * 1024) > 10.:
        #     os.kill(os.getpid(), signal.SIGKILL)
        fitness_dict = {i: per_topo_pop[i][-1] for i in per_topo_pop}
        for label in fitness_dict:
            cross_over = random.random()
            if cross_over <= recombination_rate:
                cnt = 0
                while True:
                    try:
                        candidate_keys = list(per_topo_pop.keys())
                        fitness_target = fitness_dict[label]
                        # mutate
                        candidate_keys.remove(label)
                        x_1, x_2, x_3 = random.sample(candidate_keys, 3)
                        x_1, x_2, x_3 = per_topo_pop[x_1][0], per_topo_pop[x_2][0], per_topo_pop[x_3][0]
                        candidate = x_3.deepcopy()

                        diff = differential_circuits(x_1, x_2, candidate)
                        for k in diff:
                            candidate.alter_component_value(k, diff[k])
                        with contextlib.redirect_stderr(None):
                            fitness_candidate = simulate_amp(candidate)
                        if fitness_candidate < fitness_target:
                            label_param_candidate = encode_circuit(candidate, True)
                            per_topo_pop.pop(label)
                            per_topo_pop.update({label_param_candidate: [candidate, fitness_candidate]})
                            if fitness_candidate < 1e-3:
                                break
                        else:
                            pass
                        break
                    except PySpice.Spice.NgSpice.Shared.NgSpiceCommandError:
                        cnt += 1
                        if cnt >= 10:
                            break
    # reduce the number of circuits in each topology. Default to top 1
    per_topo_pop = select_population(per_topo_pop)

    path = './checkpoint/amplifier/param_opt_4/'
    if not os.path.isdir(path):
        os.mkdir(path)
    save_circuit(per_topo_pop, path=path)


def parameter_opt(population, bar):
    """perform parameter optimization for each topology"""
    s = time.time()

    label_topo = encode_circuit(population[list(population.keys())[0]][0], False)
    parameter_worker(population, label_topo, bar)

    print(f'totally consume:{time.time() - s} to optimize topology {label_topo}')


def main(path, bar):
    """resume POM based on a evolved topology"""
    population = {}

    #  load netlists
    netlists = os.listdir(path)
    for netlist in netlists:
        netlist_path = path + '/' + netlist
        circuit = AdvanceCircuit(netlist_path)
        label_param = encode_circuit(circuit, True)
        fitness = simulate_amp(circuit)
        population.update({label_param: [circuit, fitness]})

    parameter_opt(population, bar)


if __name__ == '__main__':
    args = get_args()
    main(args.path, args.bar)

