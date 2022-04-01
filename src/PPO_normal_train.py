from argparse import ArgumentParser
from  RL_Agent.test_RL_Agent import test_RL
from RL_Agent.train_RL_Agent import train_RL
from rl_environment import  Custom_Environment
import torch
import utils

# set                                                                                                                            to cpu or cuda
device = torch.device('cpu')

if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")


parser = ArgumentParser()
parser.add_argument("-train_episodes", type = int, default=100, help="train episodes")
parser.add_argument("-test_episodes", type = int, default=10, help="test episodes")
parser.add_argument("-env_name", default="helpdesk", help="environment name")
parser.add_argument("-exp_name", default="no_action_mask_case", help="experiment name")
# parser.add_argument("-", default="no_action_mask_case", help="experiment name")

opt = parser.parse_args()

env_name = opt.env_name
# exp_name = "without_per_act_constant_end_rew"
# exp_name = "const_end_reward_no_act_loss"
# exp_name = "last_2_act_and_end_reward"
exp_name = opt.exp_name

###train 
checkpoint_dir, log_dir = utils.checkpoint_log_paths(env_name)

dname = "dataset/preprocessed/"+env_name+"_d2_"
train_data = dname+'train_RL.pkl'
val_data = dname+'val_RL.pkl'
test_data = dname+'test_RL.pkl'
mae = 1.5
# setting threshold of cumulative case KPI of a process for goal satisfaction : currently third quartile value of dataset
thresh = 13.89       #for helpdesk
if opt.env_name == "bpi_12_w":
    thresh = 18.28   
if opt.env_name == "traffic_ss":
    thresh = 607.04
else:
    thresh = utils.get_third_quartile(opt.env_name)

action_mask = True
env = Custom_Environment(train_data, env_name = env_name, mode='verbose', mae=1.5, thresh=thresh, reward_type=opt.exp_name)

if exp_name ==  "no_action_mask_case":
    action_mask = False




# ##val
# env = Custom_Environment(val_data, env_name = env_name, mode='verbose',mae=1.5, thresh=thresh,reward_type=opt.exp_name)
# test_RL(env,opt.test_episodes, checkpoint_path, log_dir,env_name,exp_name,mode="val")

# ##test
# env = Custom_Environment(test_data, env_name = env_name, mode='verbose',mae=1.5, thresh=thresh,reward_type=opt.exp_name)
# test_RL(env,opt.test_episodes, checkpoint_path, log_dir,env_name,exp_name,mode="test")

