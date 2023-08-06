import subprocess
import multiprocessing
import os
import time
import psutil
import signal

root = './checkpoint/amplifier/topo_opt_1'
cpu = multiprocessing.cpu_count()
file_list = os.listdir(root)

optimized_file = os.listdir('./checkpoint/amplifier/param_opt_1')
optimized_file = [x.split('-')[0] for x in optimized_file]

processes = []
i = 0
for file in file_list:
    if file not in optimized_file:
        path = root + '/' + file
        p = subprocess.Popen(['python', 'amplifier_param.py', f'--path={path}', f'--bar={i}'])
        processes.append(p)
        i += 1
        if psutil.cpu_percent(30) > 80.:
            time.sleep(7200)


while True:
    time.sleep(60)
    cnt = 0
    for p in processes:
        if p.poll() is not None:
            os.kill(p.pid, signal.SIGTERM)
            cnt += 1
    if cnt == len(file_list):
        break

