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
FLAG = 'Tuple1K'

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

def evaluate(position, col_idx, env):
    total_cost = 0
    tasks = []
    until = 0
    launched_task_cnt = 0

    for task_id, machine_id in enumerate(position):
        #retireve task 
        task_data = data_np[task_id]
        src_name = list(env.scenario.node_id2name.values())[0]
        # Extract relevant fields from task_data using col_idx for efficiency
        generated_time = task_data[col_idx['GenerationTime']]
       
    
        task = Task(
            id=task_id,
            task_size=task_data[col_idx['TaskSize']],
            cycles_per_bit=task_data[col_idx['CyclesPerBit']],
            trans_bit_rate=task_data[col_idx['TransBitRate']],
            ddl=task_data[col_idx['DDL']],
            src_name=src_name,
            task_name=task_data[col_idx['TaskName']],
        )

        dst_name = env.scenario.node_id2name[position[task_id]]

        tasks.append(task)
        while True:
            # Catch completed task information.
            while env.done_task_info:
                item = env.done_task_info.pop(0)
            
            if env.now >= generated_time:
                env.process(task=task, dst_name=dst_name)
                launched_task_cnt += 1
                break

            # Execute the simulation with error handler.
            try:
                env.run(until=until)
            except Exception as e:
                pass

            until += 1

    # Continue the simulation until the last task successes/fails.
    while env.task_count < launched_task_cnt:
        until += 1
        try:
            env.run(until=until)
        except Exception as e:
            pass

    #total_cost = sum(task.exe_energy for task in tasks)
    success_rate = SuccessRate().eval(env.logger.task_info)
    avg_latency = AvgLatency().eval(env.logger.task_info)
    env.reset()
    env.scenario.reset()
    
    # Higher is better: prioritize success rate, then energy (lower is better), then latency (lower is better)
    # Normalize and weight so that success rate dominates, then energy, then latency
    # All scores are positive and higher is better
    norm_success = success_rate  # already in [0,1]
    norm_energy = 1 / (1 + total_cost)  # lower energy -> closer to 1
    norm_latency = 1 / (1 + avg_latency)  # lower latency -> closer to 1

    # Weights: success rate is most important, then energy, then latency
    score = (norm_success * 1e3) +  (norm_latency * 1e2) #+ (norm_energy * 1e6)
    return score

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
        self.score = evaluate(self.position,self.col_idx, self.env)

    def update_velocity(self, global_best, w=0.7, c1=2, c2=2):
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
            v = max(-2, min(2, v))  # Allow a wider range for velocity
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

def create_particle(flag, data_np, col_idx, _):
    # Use shared data_np and col_idx, do not reload
    return Particle(flag, data_np, col_idx)

# Global cache for data in each worker process
_DATA_CACHE = {}

def eval_worker(args):
    try:
        position,scenario_config, flag = args
        # global _DATA_CACHE
        # if data_path not in _DATA_CACHE:
        #     _DATA_CACHE[data_path] = pd.read_csv(data_path)
        # data = _DATA_CACHE[data_path]
        # data_np = data.to_numpy()
        scenario = Scenario(config_file=scenario_config, flag=flag)
        env = Env(scenario, config_file="core/configs/env_config_null.json", enable_logging=False)
        return evaluate(position, col_idx, env)
    except Exception as e:
        import traceback, os
        print(f"Exception in eval_worker (pid={os.getpid()}): {e}")
        with open(f'/tmp/eval_worker_error_{os.getpid()}.log', 'w') as f:
            f.write("Exception in eval_worker:\n")
            traceback.print_exc(file=f)
        return float('-inf')

def run_pso(
    population_size=50,
    max_iterations=50,
    w=0.7,
    c1=2.0,
    c2=2.0,
    flag=FLAG,
    config_path=CONFIG_PATH,
    data_np=data_np,
    col_idx=col_idx,
    enable_logging=True
):
    """
    Run PSO with custom hyperparameters and return best score and solution.
    Args:
        population_size (int): Number of particles.
        max_iterations (int): Number of PSO epochs.
        w (float): Inertia weight.
        c1 (float): Cognitive coefficient.
        c2 (float): Social coefficient.
        flag (str): Dataset/scenario flag.
        config_path (str): Path to scenario config.
        data_np (np.ndarray): Task dataset as numpy array.
        col_idx (dict): Column index mapping.
        enable_logging (bool): Whether to print progress/logging info.
    Returns:
        dict: {'best_position': ..., 'best_score': ..., 'avg_scores': ..., 'best_scores': ..., 'min_scores': ..., 'max_scores': ...}
    """
    # Prepare swarm
    swarm = [
        create_particle(flag, data_np, col_idx, i)
        for i in tqdm(range(population_size), desc="Initializing swarm")
    ]
    best_particle = max(swarm, key=lambda p: p.score)
    global_best = best_particle.best_position
    global_best_score = best_particle.score
    scenario_eval = Scenario(config_file=config_path, flag=flag)
    env_eval = Env(scenario_eval, config_file="core/configs/env_config_null.json", enable_logging=False)
    global_best_score = evaluate(global_best, col_idx, env_eval)
    avg_scores = []
    best_scores = []
    min_scores = []
    max_scores = []
    iterator = trange(max_iterations, desc="PSO Progress") if enable_logging else range(max_iterations)
    for iteration in iterator:
        for particle in swarm:
            particle.update_velocity(global_best, w=w, c1=c1, c2=c2)
        eval_args = [(particle.position[:], config_path, flag) for particle in swarm]
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            scores = list(executor.map(eval_worker, eval_args))
        for idx, particle in enumerate(swarm):
            particle.update_position()
            score = scores[idx]
            if score > particle.score:
                particle.best_position = particle.position[:]
                particle.score = score
        best_particle = max(swarm, key=lambda p: p.score)
        if best_particle.score > global_best_score:
            global_best = best_particle.best_position[:]
            global_best_score = best_particle.score
        avg_score = sum(p.score for p in swarm) / len(swarm)
        min_score = min(p.score for p in swarm)
        max_score = max(p.score for p in swarm)
        avg_scores.append(avg_score)
        best_scores.append(global_best_score)
        min_scores.append(min_score)
        max_scores.append(max_score)
    return {
        'best_position': global_best,
        'best_score': global_best_score,
        'avg_scores': avg_scores,
        'best_scores': best_scores,
        'min_scores': min_scores,
        'max_scores': max_scores
    }

# Exécution
if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()
    result = run_pso()
    profiler.disable()
    solution = result['best_position']
    cost = result['best_score']
    avg_scores = result['avg_scores']
    best_scores = result['best_scores']
    min_scores = result['min_scores']
    max_scores = result['max_scores']
    print("Meilleure solution :", solution)
    print("Coût associé :", cost)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,6))
    plt.plot(avg_scores, label='Average Score', color='tab:blue')
    plt.plot(best_scores, label='Best Score', color='tab:orange')
    plt.fill_between(range(len(avg_scores)), min_scores, max_scores, color='gray', alpha=0.2, label='Score Range (min-max)')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('PSO Score Evolution Across Epochs')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('pso_score_evolution.png')
    print("Plot saved as pso_score_evolution.png")
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats(30)  # Print top 30 functions by cumulative time
