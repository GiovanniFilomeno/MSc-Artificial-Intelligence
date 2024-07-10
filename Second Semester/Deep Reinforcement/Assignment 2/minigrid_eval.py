# imports
import argparse
import os
import numpy as np
import random

import gymnasium as gym

import torch
from torch.distributions.categorical import Categorical

import onnx
from onnx2pytorch import ConvertModel
from minigrid.wrappers import ImgObsWrapper


# Seed random number generators
torch.backends.cudnn.deterministic = True
if os.path.exists("seed.rnd"):
    with open("seed.rnd", "r") as f:
        seed = int(f.readline().strip())
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
else:
    seed = None
# print(f"Seed: {seed}")

class ChannelFirst(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        old_shape = env.observation_space.shape
        self.observation_space = {}
        self.observation_space = gym.spaces.Box(0, 255, shape=(3, 7, 7))

    def observation(self, observation):
        return np.swapaxes(observation, 2, 0)

class ScaledFloatFrame(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)

    def observation(self, observation):
        # careful! This undoes the memory optimization, use
        # with smaller replay buffers only.
        return np.array(observation).astype(np.float32)

class MinigridDoorKey6x6ImgObs(gym.Wrapper):
    """Minigrid with image observations provided by minigrid, partially observable."""
    def __init__(self, render=False):
        if render:
          env = gym.make('MiniGrid-DoorKey-6x6-v0', render_mode="rgb_array")
        else:
          env = gym.make('MiniGrid-DoorKey-6x6-v0')
        env = ScaledFloatFrame(ChannelFirst(ImgObsWrapper(env)))
        super().__init__(env)


class Agent():
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def select_action(self, state):
        state = np.expand_dims(state, 0)
        state = torch.tensor(state, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            q_action = self.model(state).detach().cpu().numpy()

            return np.argmax(q_action)

def run_episode(agent, seed=None):
    env = MinigridDoorKey6x6ImgObs()
    state = env.reset(seed=seed)[0]
    score = 0
    done = False
    while not done:
        action = agent.select_action(state)
        state, reward, terminated, truncated, _ = env.step(action)
        score += reward
        done = terminated or truncated
    env.close()
    return score


if __name__ == "__main__":
    N_EPISODES = 50

    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", type=str)
    args = parser.parse_args()
    model_file = args.submission

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = ConvertModel(onnx.load(model_file))
    model.eval()
    model = model.to(device)
    agent = Agent(model=model, device=device)

    # Evaluate model
    scores = []
    for i in range(N_EPISODES):
        if seed is not None:
            seed = np.random.randint(1e7)
        scores.append(run_episode(agent, seed=seed))

    # Print result
    print(np.mean(scores))