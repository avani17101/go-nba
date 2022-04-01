from rl_environment_base2 import Env_base
import utils
from tensorforce.environments import Environment
from tensorforce.agents import Agent
import numpy as np
import torch.nn.functional as F
import torch
from tqdm import tqdm
from statistics import mean
import csv

device = torch.device('cpu')

if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")

env_name = "helpdesk"
exp_name = "PPO_normal"
dname = "dataset/preprocessed/"+env_name+"_d2_"
train_data = dname+'train_RL.pkl'
val_data = dname+'val_RL.pkl'
test_data = dname+'test_RL.pkl'
mae = 1.5
# setting threshold of cumulative case KPI of a process for goal satisfaction : currently third quartile value of dataset
thresh = 13.89       #for helpdesk
if env_name == "bpi_12_w":
    thresh = 18.28   
if env_name == "traffic_ss":
    thresh = 607.04
else:
    thresh = utils.get_third_quartile(env_name)

environment = Environment.create(
    environment= Env_base(train_data, env_name = env_name, mode='verbose', mae=1.5, thresh=thresh, reward_type=exp_name), max_episode_timesteps=800000
)

from tensorforce.agents import Agent
from tensorforce.execution import Runner
agent = Agent.create(
    agent='ppo', environment=environment, batch_size=10, learning_rate=1e-3
)


reward_lis = []
fields = ["episode","reward","accuracy","mae","gs","gv","gv_turned_gs","gs_turned_gv"]
rows = []
log_path = "tensorforce_logs/ppo.csv"
csvfile = open(log_path, 'w') 
# creating a csv writer object
csvwriter = csv.writer(csvfile)
# writing the fields
csvwriter.writerow(fields)
episodes = 1000
for i in tqdm(range(episodes)):
    states = environment.reset()
    terminal = False
    reward_epi = []
    while not terminal:
        actions = agent.act(states=states)
        states, reward, terminal, _ = environment.step(actions)
        reward_epi.append(reward)
        agent.observe(terminal=terminal, reward=reward)
    avg_reward = mean(reward_epi)
    environment.render()
    rows.append([i, avg_reward, environment.accuracy_final, environment.mae_final, environment.percent_overall_gs, environment.percent_overall_gv, environment.percent_gv_which_became_gs,environment.percent_gs_which_became_gv])
    
agent.save("RL_checkpoints/helpdeskPPO")
csvwriter.writerows(rows)
csvfile.close()

    
