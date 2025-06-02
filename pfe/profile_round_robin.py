import sys
import os
# Add project root to sys.path for module imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import cProfile
import pstats
from core.env import Env
from examples.scenarios.scenario_1 import Scenario
from policies.demo.demo_round_robin import DemoRoundRobin

def run_simulation():
    config_path = os.path.join(os.path.dirname(__file__), '../examples/scenarios/configs/config_1.json')
    scenario = Scenario(config_file=config_path)
    env = Env(scenario, config_file=config_path)
    policy = DemoRoundRobin()
    env.policy = policy  # Set the policy in the environment
    env.run(until=10000)  # Run the simulation for a fixed time

def main():
    profiler = cProfile.Profile()
    profiler.enable()
    run_simulation()
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.dump_stats('profile_round_robin.prof')
    with open('profile_round_robin.txt', 'w') as f:
        stats.stream = f
        stats.sort_stats('cumtime').print_stats(150)  # Print top 50 by cumulative time
    print('Profiling complete. Results saved to profile_round_robin.prof and profile_round_robin.txt')

if __name__ == "__main__":
    main()
