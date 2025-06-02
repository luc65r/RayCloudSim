import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from pso import run_pso
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger("PSO-DRL")

class PSOHparamEnv(gym.Env):
    """
    Gymnasium environment for PSO hyperparameter optimization.
    Observation: dummy (no state, stateless)
    Action: [w, c1, c2] (continuous)
    Reward: best PSO score (higher is better)
    """
    def __init__(self):
        super().__init__()
        # Action space: w in [0.4, 1.2], c1 in [1.0, 3.0], c2 in [1.0, 3.0]
        self.action_space = gym.spaces.Box(
            low=np.array([0.4, 1.0, 1.0]),
            high=np.array([1.2, 3.0, 3.0]),
            dtype=np.float32
        )
        # Observation space: dummy (stateless)
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(1,), dtype=np.float32
        )
        self.last_score = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        logger.info("Environment reset.")
        return np.array([0.0], dtype=np.float32), {}

    def step(self, action):
        w, c1, c2 = float(action[0]), float(action[1]), float(action[2])
        logger.info(f"Step with action: w={w:.3f}, c1={c1:.3f}, c2={c2:.3f}")
        # Run PSO with these hyperparameters (small pop/iters for speed)
        result = run_pso(
            population_size=20,
            max_iterations=10,
            w=w,
            c1=c1,
            c2=c2,
            enable_logging=False
        )
        reward = result['best_score']
        self.last_score = reward
        logger.info(f"PSO run complete. Reward (best_score): {reward:.4f}")
        # Stateless, so always done after one step
        obs = np.array([0.0], dtype=np.float32)
        terminated = True
        truncated = False
        info = {'score': reward}
        return obs, reward, terminated, truncated, info

if __name__ == "__main__":
    # Optional: check the environment
    env = PSOHparamEnv()
    check_env(env, warn=True)

    # Train PPO agent to find best PSO hyperparameters
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=50)  # Increase for real search

    # Test the trained agent
    obs, _ = env.reset()
    action, _ = model.predict(obs, deterministic=True)
    logger.info(f"Best hyperparameters found by PPO: {action}")
    obs, reward, terminated, truncated, info = env.step(action)
    logger.info(f"Best PSO score found: {reward}")
