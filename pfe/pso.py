import os
import sys
import json
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import partial
import cProfile
import pstats

current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import random
from eval.benchmarks.Pakistan.scenario import Scenario
from eval.metrics.metrics import SuccessRate, AvgLatency
from core.env import Env
from core.task import Task
import pandas as pd
from tqdm import trange
from tqdm import tqdm

# Paramètres
FLAG = 'Tuple30K'

# Load config to determine number of machines and tasks
def get_machine_and_task_count(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    nodes = config['Nodes']
    # Assume the first node is the input node (e.g., 'e0')
    num_machines = len(nodes) - 1
    return num_machines, num_machines

CONFIG_PATH = f"eval/benchmarks/Pakistan/data/{FLAG}/config.json"
NUM_MACHINES, NUM_TASKS = get_machine_and_task_count(CONFIG_PATH)

data = pd.read_csv(f"eval/benchmarks/Pakistan/data/{FLAG}/testset.csv")
data_np = data.to_numpy()
col_idx = {name: i for i, name in enumerate(data.columns)}

NUM_TASKS = len(data)  # Dimension of the search space = number of tasks in the dataset
POPULATION_SIZE = 50   # Fixed number of particles
MAX_ITERATIONS = 50    # Nombre d'itérations

def evaluate(position, data_np, col_idx, env):
    total_cost = 0
    tasks = []
    test_tasks = []
    for task_id, machine_id in enumerate(position):
        task_data = data_np[task_id]
        src_name = list(env.scenario.node_id2name.values())[0]
        test_tasks.append([
            f"Task{task_id}",
            0,
            task_id,
            task_data[col_idx['TaskSize']],
            task_data[col_idx['CyclesPerBit']],
            task_data[col_idx['TransBitRate']],
            task_data[col_idx['DDL']],
            src_name,
            env.scenario.node_id2name[machine_id]
        ])
    for task_info in test_tasks:
        generated_time = task_info[1]
        task = Task(
            id=task_info[2],
            task_size=task_info[3],
            cycles_per_bit=task_info[4],
            trans_bit_rate=task_info[5],
            ddl=task_info[6],
            src_name=task_info[7],
            task_name=task_info[0],
        )
        dst_name = env.scenario.node_id2name[position[task_info[2]]]
        dst_node = env.scenario.get_node(dst_name)
        tasks.append(task)
        try:
            env.process(task=task, dst_name=dst_node.name)
        except Exception:
            pass
    try:
        env.run(until=100000)
    except Exception:
        pass
    total_cost = sum(task.exe_energy for task in tasks)
    success_rate = SuccessRate().eval(env.logger.task_info)
    avg_latency = AvgLatency().eval(env.logger.task_info)
    env.reset()
    env.scenario.reset()
    return total_cost + (1 - success_rate) * 1000 + avg_latency * 10
        

class Particle:
    """
    Représente une particule dans l'algorithme PSO.
    Chaque particule a une position (solution actuelle), une vitesse, et garde en mémoire sa meilleure position.
    """

    def __init__(self, flag, data_np, col_idx):
        """
        Initialise une particule avec une position et une vitesse aléatoires.
        Chaque particule a son propre Scenario et Env pour éviter les problèmes de thread-safety.
        """
        self.position = [random.randint(0, NUM_MACHINES - 1) for _ in range(NUM_TASKS)]
        self.velocity = [random.randint(-1, 1) for _ in range(NUM_TASKS)]
        self.best_position = self.position[:]
        self.data_np = data_np  # Shared reference
        self.col_idx = col_idx  # Shared reference
        # Each particle gets its own scenario/env
        self.scenario = Scenario(config_file=f"eval/benchmarks/Pakistan/data/{flag}/config.json", flag=flag)
        self.env = Env(self.scenario, config_file="core/configs/env_config_null.json", enable_logging=False)
        self.best_score = evaluate(self.position, self.data_np, self.col_idx, self.env)

    def update_velocity(self, global_best, w=0.5, c1=1, c2=1):
        """
        Met à jour la vitesse de la particule selon la formule PSO.

        Args:
            global_best (list of int): Meilleure position globale connue.
            w (float): Poids d'inertie.
            c1 (float): Coefficient cognitif.
            c2 (float): Coefficient social.
        """
        new_velocity = []
        for i in range(NUM_TASKS):
            r1, r2 = random.random(), random.random()
            cognitive = c1 * r1 * (self.best_position[i] - self.position[i])
            social = c2 * r2 * (global_best[i] - self.position[i])
            v = int(w * self.velocity[i] + cognitive + social)
            v = max(-1, min(1, v))  # Limite la vitesse entre -1 et 1
            new_velocity.append(v)
        self.velocity = new_velocity

    def update_position(self):
        """
        Met à jour la position de la particule selon sa vitesse.
        Évalue ensuite la nouvelle position et met à jour le meilleur personnel si nécessaire.
        """
        new_position = []
        for i in range(NUM_TASKS):
            val = (self.position[i] + self.velocity[i]) % NUM_MACHINES
            new_position.append(val)
        self.position = new_position

        score = evaluate(self.position, self.data_np, self.col_idx, self.env)
        if score < self.best_score:
            self.best_position = self.position[:]
            self.best_score = score

def create_particle(flag, data_np, col_idx, _):
    # Use shared data_np and col_idx, do not reload
    return Particle(flag, data_np, col_idx)

# Global cache for data in each worker process
_DATA_CACHE = {}

def eval_worker(args):
    try:
        position, data_path, scenario_config, flag = args
        global _DATA_CACHE
        if data_path not in _DATA_CACHE:
            _DATA_CACHE[data_path] = pd.read_csv(data_path)
        data = _DATA_CACHE[data_path]
        data_np = data.to_numpy()
        col_idx = {name: i for i, name in enumerate(data.columns)}
        scenario = Scenario(config_file=scenario_config, flag=flag)
        env = Env(scenario, config_file="core/configs/env_config_null.json", enable_logging=False)
        return evaluate(position, data_np, col_idx, env)
    except Exception as e:
        import traceback, os
        with open(f'/tmp/eval_worker_error_{os.getpid()}.log', 'w') as f:
            f.write("Exception in eval_worker:\n")
            traceback.print_exc(file=f)
        return float('inf')

def pso():
    """
    Exécute l'algorithme PSO pour optimiser l'affectation des tâches aux machines.

    Returns:
        tuple: (meilleure solution trouvée, score associé)
    """
    data_path = f"eval/benchmarks/Pakistan/data/{FLAG}/testset.csv"
    config_file_path = f"eval/benchmarks/Pakistan/data/{FLAG}/config.json"
    # Load data and col_idx ONCE and share
    data = pd.read_csv(data_path)
    data_np = data.to_numpy()
    col_idx = {name: i for i, name in enumerate(data.columns)}
    # Parallelize swarm initialization using ThreadPoolExecutor to avoid pickling issues
    create_particle_partial = partial(create_particle, FLAG, data_np, col_idx)
    with ThreadPoolExecutor() as executor:
        swarm = list(tqdm(executor.map(create_particle_partial, range(POPULATION_SIZE)), total=POPULATION_SIZE, desc="Initializing swarm"))
    global_best = min(swarm, key=lambda p: p.best_score).best_position
    # Use a new scenario/env for global_best evaluation to avoid thread issues
    scenario_eval = Scenario(config_file=f"eval/benchmarks/Pakistan/data/{FLAG}/config.json", flag=FLAG)
    env_eval = Env(scenario_eval, config_file="core/configs/env_config_null.json", enable_logging=False)
    global_best_score = evaluate(global_best, data_np, col_idx, env_eval)
    avg_scores = []
    for iteration in trange(MAX_ITERATIONS, desc="PSO Progress"):
        # Update velocities in main thread (fast, in-memory)
        for particle in swarm:
            particle.update_velocity(global_best)
        # Prepare evaluation arguments for parallel processing
        eval_args = [(particle.position[:], data_path, config_file_path, FLAG) for particle in swarm]
        # Evaluate new positions in parallel using processes
        with ProcessPoolExecutor(max_workers=4) as executor:  # Limit to 4 workers to avoid OOM
            scores = list(executor.map(eval_worker, eval_args))
        # Update particles with new positions and scores
        for idx, particle in enumerate(swarm):
            particle.update_position()  # update position in main thread
            score = scores[idx]
            if score < particle.best_score:
                particle.best_position = particle.position[:]
                particle.best_score = score
        best_particle = min(swarm, key=lambda p: p.best_score)
        if best_particle.best_score < global_best_score:
            global_best = best_particle.best_position[:]
            global_best_score = best_particle.best_score
        avg_score = sum(p.best_score for p in swarm) / len(swarm)
        avg_scores.append(avg_score)
    return global_best, global_best_score

# Exécution
if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()
    solution, cost = pso()
    profiler.disable()
    print("Meilleure solution :", solution)
    print("Coût associé :", cost)
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats(30)  # Print top 30 functions by cumulative time
