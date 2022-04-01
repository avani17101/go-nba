from rl_base_tf import Env_base
from tensorforce.environments import Environment
from tensorforce.agents import Agent
import numpy as np
import torch.nn.functional as F
import torch
from tqdm import tqdm
from statistics import mean
import csv
from tensorforce.agents import Agent
from argparse import ArgumentParser
import os

device = torch.device('cpu')

if(torch.cuda.is_available()): 
    device = torch.device('cuda') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")


def train(agent, log_path, save_path, environment,episodes):
    fields = ["episode","reward","accuracy_last","accuracy_last2","accuracy_last3","gs","gv","gv_turned_gs","gs_turned_gv"]
    rows = []
    csvfile = open(log_path, 'w+') 
    # creating a csv writer object
    csvwriter = csv.writer(csvfile)
    # writing the fields
    csvwriter.writerow(fields)
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
        rows.append([i, avg_reward, np.mean(environment.accuracy_last1), np.mean(environment.accuracy_last2), np.mean(environment.accuracy_last3), environment.percent_overall_gs, environment.percent_overall_gv, environment.percent_gv_which_became_gs,environment.percent_gs_which_became_gv])
    agent.save(save_path)
    csvwriter.writerows(rows)
    csvfile.close()


# evaluate agent on val and test datasets
def eval(agent, log_path,environment,episodes):
    fields = ["episode","compliance","reward","accuracy_last","accuracy_last2","accuracy_last3","gs","gv","gv_turned_gs","gs_turned_gv"]
    rows = []
    csvfile = open(log_path, 'w') 
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(fields)
    for i in tqdm(range(episodes)):
        states = environment.reset()
        internals = agent.initial_internals()
        terminal = False
        reward_epi = []
        while not terminal:
            actions, internals = agent.act(
                states=states, internals=internals,
                independent=True, deterministic=True
            )
            states, reward, terminal, _  = environment.step(actions=actions)
            reward_epi.append(reward)
        avg_reward = mean(reward_epi)
        environment.render()
        rows.append([i, environment.compliance_per, avg_reward, np.mean(environment.accuracy_last1), np.mean(environment.accuracy_last2), np.mean(environment.accuracy_last3), environment.percent_overall_gs, environment.percent_overall_gv, environment.percent_gv_which_became_gs,environment.percent_gs_which_became_gv])
    csvwriter.writerows(rows)
    csvfile.close()
    return


#***************************************************meta data***********************************************************************
parser = ArgumentParser()
parser.add_argument("--env_name", default="helpdesk", help="dataset name")
parser.add_argument("--agent_type", default="ppo", help="agent type")
parser.add_argument("--train_epi", default=10, type = int, help="train episodes")
parser.add_argument("--test_epi", default=2, type = int, help="test episodes")
#***************************************************close***********************************************************************
opt = parser.parse_args()
agent_type = opt.agent_type
env_name = opt.env_name
reward_type = "last_3_act_and_end_reward"
dname = "dataset/preprocessed/"+env_name+"_d2_"
train_data = dname+'train_RL.pkl'
test_data = dname+'test_RL.pkl'
# setting threshold of cumulative case KPI of a process for goal satisfaction : currently third quartile value of dataset
thresh = 13.89   #for helpdesk
if env_name == "bpi_12_w":
    thresh = 18.28   
if env_name == "traffic_ss":
    thresh = 607.04
if env_name == "bpi2019":
    thresh = 94.015

# training agent 
environment = Environment.create(
    environment= Env_base(train_data, env_name = env_name, mode='verbose', mae=1.5, thresh=thresh, reward_type=reward_type), max_episode_timesteps=800000
)
agent = None
if agent_type == 'ppo':
    agent = Agent.create(
        agent=agent_type, environment=environment, batch_size=64, learning_rate=1e-2,config = {"device":'GPU'}
    ) 
if agent_type == 'dqn':
    agent = Agent.create(
        agent='dqn', environment=environment, batch_size=64, memory=900000,learning_rate=1e-2,config = {"device":'GPU'}
    )
exp_name = "negfinal"
path = "/ssd_scratch/cvit/avani.gupta/"
save_path = "tensorforce_logs/"+agent_type+env_name+reward_type+exp_name
if not os.path.isdir(save_path):
    os.makedirs(save_path)

log_path = "tensorforce_logs/"+agent_type+env_name+reward_type+exp_name+"train.csv"
train(agent, log_path, save_path, environment, opt.train_epi)
# Close agent and environment
agent.close()
environment.close()


# testing agent
log_path = "tensorforce_logs/"+agent_type+env_name+reward_type+exp_name+"test.csv"
environment = Environment.create(
    environment= Env_base(test_data, env_name = env_name, mode='verbose', mae=1.5, thresh=thresh, reward_type=reward_type), max_episode_timesteps=800000
)
agent = Agent.load(save_path, "agent", "checkpoint", environment)
eval(agent, log_path, environment, opt.test_epi)
# Close agent and environment
agent.close()
environment.close()





    
