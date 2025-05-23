import os
import sys

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
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Paramètres
NUM_TASKS = 3           # Nombre de tâches
NUM_MACHINES = 4        # Nombre de machines (0, 1, 2, 3)
POPULATION_SIZE = 10    # Nombre de particules
MAX_ITERATIONS = 50     # Nombre d'itérations
FLAG = 'Tuple30K'

def evaluate(position,data,env):
    """
    Évalue une position (solution candidate) dans l'espace de recherche.

    Args:
        position (list of int): Liste des machines assignées à chaque tâche.

    Returns:
        int: Score de la solution (plus petit est meilleur). Ici, on minimise la somme.
    """

    for i in range(NUM_TASKS):
        task_info = data.iloc[i]
        generated_time = task_info['GenerationTime']
        task = Task(
            id=task_info['TaskID'],
            task_size=task_info['TaskSize'],
            cycles_per_bit=task_info['CyclesPerBit'],
            trans_bit_rate=task_info['TransBitRate'],
            ddl=task_info['DDL'],
            src_name='e0',
            task_name=task_info['TaskName'],
        )

        max_wait = 10000  # Prevent infinite loop by limiting iterations
        wait_count = 0
        while True:
            # Catch completed task information.
            while env.done_task_info:
                env.done_task_info.pop(0)
            
            if env.now >= generated_time:
                dst_id = position[i] # offloading decision
                dst_name = env.scenario.node_id2name[dst_id]
                env.process(task=task, dst_name=dst_name)
                break

            wait_count += 1
            if wait_count > max_wait:
                logging.warning(f"Breaking out of infinite loop for task {task_info['TaskID']}")
                break
    

class Particle:
    """
    Représente une particule dans l'algorithme PSO.
    Chaque particule a une position (solution actuelle), une vitesse, et garde en mémoire sa meilleure position.
    """

    def __init__(self,env,data):
        """
        Initialise une particule avec une position et une vitesse aléatoires.
        """
        self.position = [random.randint(0, NUM_MACHINES - 1) for _ in range(NUM_TASKS)]
        self.velocity = [random.randint(-1, 1) for _ in range(NUM_TASKS)]
        self.best_position = self.position[:]
        self.env = env
        self.data = data
        self.best_score = evaluate(self.position,self.data,self.env)

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

        score = evaluate(self.position,self.data,self.env)
        if score < self.best_score:
            self.best_position = self.position[:]
            self.best_score = score

def pso():
    """
    Exécute l'algorithme PSO pour optimiser l'affectation des tâches aux machines.

    Returns:
        tuple: (meilleure solution trouvée, score associé)
    """
    logging.info("Chargement du scénario...")
    scenario = Scenario(config_file=f"eval/benchmarks/Pakistan/data/{FLAG}/config.json", flag=FLAG)

    logging.info("Initialisation de l'environnement...")
    env = Env(scenario, config_file="core/configs/env_config_null.json", enable_logging=True)

    logging.info("Chargement des données de test...")
    data = pd.read_csv(f"eval/benchmarks/Pakistan/data/{FLAG}/testset.csv")

    logging.info("Initialisation du swarm de particules...")
    swarm = []
    for i in tqdm(range(POPULATION_SIZE), desc="Initializing swarm"):
        swarm.append(Particle(env, data))
        tqdm.write(f"Particle {i+1}/{POPULATION_SIZE} initialized")

    logging.info("Évaluation initiale du meilleur global...")
    global_best = min(swarm, key=lambda p: p.best_score).best_position
    global_best_score = evaluate(global_best, data, env)

    logging.info("Début des itérations PSO...")
    for iteration in trange(MAX_ITERATIONS, desc="PSO Progress"):
        for particle in swarm:
            particle.update_velocity(global_best)
            particle.update_position()

        best_particle = min(swarm, key=lambda p: p.best_score)
        if best_particle.best_score < global_best_score:
            global_best = best_particle.best_position[:]
            global_best_score = best_particle.best_score

    logging.info("Optimisation terminée.")
    logging.info(f"Meilleure solution trouvée : {global_best}, coût : {global_best_score}")
    return global_best, global_best_score

# Exécution
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Démarrage de l'algorithme PSO...")
    solution, cost = pso()
    print("Meilleure solution :", solution)
    print("Coût associé :", cost)
