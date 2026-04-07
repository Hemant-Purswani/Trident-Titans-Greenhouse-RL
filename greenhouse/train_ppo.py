import os
import sys
import argparse
import gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
from pathlib import Path

# Ensure we can import from greenhouse package
sys.path.insert(0, str(Path(__file__).parent))

from server.greenhouse_environment import GreenhouseEnvironment
from models import GreenhouseAction

# ─── Gym Wrapper ─────────────────────────────────────────────────────────────

class GreenhouseGymEnv(gym.Env):
    """
    OpenAI Gym wrapper for the Greenhouse Climate Control environment.
    Allows training with Stable-Baselines3.
    """
    def __init__(self, task_id="maintain_temperature"):
        super(GreenhouseGymEnv, self).__init__()
        self.env = GreenhouseEnvironment(task_id=task_id)
        
        # Action space: 4 continuous controls [0, 1]
        self.action_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(4,), dtype=np.float32
        )
        
        # Observation space: 11 normalized values
        # [temp, humid, co2, light, out_temp, out_humid, cloud, hour, day, health, progress]
        self.observation_space = gym.spaces.Box(
            low=-2.0, high=5.0, shape=(11,), dtype=np.float32
        )

    def _get_obs(self, obs_model):
        """Normalize and vectorise the observation."""
        return np.array([
            (obs_model.temperature - 20) / 10.0,
            (obs_model.humidity - 60) / 40.0,
            (obs_model.co2_level - 1000) / 1000.0,
            obs_model.light_intensity / 1000.0,
            (obs_model.outside_temperature - 15) / 15.0,
            (obs_model.outside_humidity - 50) / 50.0,
            obs_model.cloud_cover,
            obs_model.hour_of_day / 24.0,
            obs_model.day_number / 7.0,
            obs_model.plant_health,
            obs_model.growth_progress
        ], dtype=np.float32)

    def reset(self):
        obs_model = self.env.reset()
        return self._get_obs(obs_model)

    def step(self, action_vec):
        # Convert vector to GreenhouseAction
        action = GreenhouseAction(
            heater_power=float(action_vec[0]),
            ventilation_rate=float(action_vec[1]),
            humidifier_level=float(action_vec[2]),
            artificial_lighting=float(action_vec[3])
        )
        
        obs_model = self.env.step(action)
        obs_vec = self._get_obs(obs_model)
        
        reward = float(obs_model.reward)
        done = bool(obs_model.done)
        
        # Additional info for debugging
        info = {
            "health": obs_model.plant_health,
            "growth": obs_model.growth_progress,
            "step": obs_model.step_number
        }
        
        return obs_vec, reward, done, info

    def render(self, mode="human"):
        pass

# ─── Training Script ─────────────────────────────────────────────────────────

def train(task_id, total_steps, model_name):
    print(f"🚀 Starting PPO training for task: {task_id}")
    print(f"   Steps: {total_steps} | Model: {model_name}")
    
    # Create env
    env = GreenhouseGymEnv(task_id=task_id)
    eval_env = GreenhouseGymEnv(task_id=task_id)
    
    # Check if env is valid
    # check_env(env)
    
    # Initialize model
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        device="cpu"
    )
    
    # Evaluation callback
    eval_callback = EvalCallback(
        eval_env, 
        best_model_save_path=f"./models/{task_id}_best",
        log_path=f"./logs/{task_id}", 
        eval_freq=min(5000, total_steps // 5),
        deterministic=True, 
        render=False
    )
    
    # Train
    model.learn(total_timesteps=total_steps, callback=eval_callback)
    
    # Save final model
    os.makedirs("models", exist_ok=True)
    model.save(f"models/{model_name}")
    print(f"✅ Training complete! Model saved to models/{model_name}.zip")
    
    # Final evaluation
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
    print(f"📊 Evaluation: Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    
    return model

def plot_results(task_id):
    # This is a placeholder for actual log plotting
    # In a real scenario, we'd read monitor.csv or logs
    print("📈 Generating results summary...")
    
# ─── CLI Entry ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RL Greenhouse PPO Trainer")
    parser.add_argument("--task", type=str, default="maintain_temperature", choices=TASKS)
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--output", type=str, default="ppo_greenhouse_model")
    
    # Add TASKS constant if it doesn't exist to avoid error
    TASKS = ["maintain_temperature", "optimize_growth", "weather_resilience"]
    
    args = parser.parse_args()
    
    # Start training
    train(args.task, args.steps, args.output)
