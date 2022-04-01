from tensorforce.environments import Environment
from tensorforce.agents import Agent
import numpy as np
import torch.nn.functional as F
from statistics import mean

environment = Environment.create(
    environment=HelpdeskEnv, max_episode_timesteps=100
)

agent = Agent.create(
    agent='ppo', environment=environment, batch_size=10, learning_rate=1e-3
)

# Create agent and environment
reward_lis = []
for _ in range(100):
    states = environment.reset()
    terminal = False
    reward_epi = []
    while not terminal:
        actions = agent.act(states=states)
        states, terminal, reward = environment.execute(actions=actions)
        reward_epi.append(reward)
        agent.observe(terminal=terminal, reward=reward)
    reward_lis.append(mean(reward_epi))
    
episodes = np.arange(0,len(reward_lis))
plt.plot(episodes, reward_lis)
plt.title("reward vs episodes")
plt.xlabel("episodes")
plt.ylabel("reward")
print("mean reward :",mean(reward_lis))