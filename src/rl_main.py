
from argparse import ArgumentParser
# from  RL_Agent.test_RL_Agent import test_RL
from RL_Agent.train_RL_Agent import train_RL
from rl_environment2 import  Custom_Environment
import utils
import time

parser = ArgumentParser()
parser.add_argument("-train_episodes", type = int, default=20, help="train episodes")
parser.add_argument("-test_episodes", type = int, default=1, help="test episodes")
parser.add_argument("-env_name", default="helpdesk", help="environment name")
parser.add_argument("-reward_type", default="last_1_act_and_end_reward", help="reward type")
parser.add_argument("-exp_name", default="last_1_act_and_end_reward_finalzz_dfg_pred", help="experiment name")
# parser.add_argument("-", default="no_action_mask_case", help="experiment name")

opt = parser.parse_args()
env_name = opt.env_name
exp_name = opt.exp_name
reward_type = opt.reward_type
###train 
checkpoint_dir, log_dir = utils.checkpoint_log_paths(env_name)
reward_type = opt.reward_type
dname = "dataset/preprocessed/"+env_name+"_d2_"
train_data = dname+'train_RL.pkl'
test_data = dname+'test_RL.pkl'

thresh = 13.89    # for helpdesk
if opt.env_name == "bpi_12_w":
    thresh = 24.001
if opt.env_name == "traffic_ss":
    thresh = 607.04
if opt.env_name == "bpi2019":
    thresh = 94.015

action_mask = True
env = Custom_Environment(train_data, env_name = env_name, mode='verbose', mae=1.5, thresh=thresh, reward_type=reward_type)

if exp_name ==  "no_action_mask_case":
    action_mask = False

# if agent_type == "DQN":
tic = time.time()
checkpoint_path = "RL_checkpoints/PPO_preTrained/"+env_name+"/PPO_"+env_name+"_"+exp_name+".pth"
checkpoint_path, log_path = train_RL(env, opt.train_episodes, checkpoint_dir, log_dir, env_name, exp_name, action_mask, checkpoint_path, resume_training=False)
tac = time.time()
print("total time taken:",tac-tic)
# ##test 
# env = Custom_Environment(test_data, env_name = env_name, mode='verbose',mae=1.5, thresh=thresh,reward_type= reward_type)
# test_RL(env,opt.test_episodes, checkpoint_path, log_dir,env_name,exp_name,mode="test")

