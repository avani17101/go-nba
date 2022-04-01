import pandas as pd
from rl_environment2 import  Custom_Environment
from  RL_Agent.test_RL_Agent import test_RL



#100 episodes avg
exp_names = ["last_2_act_and_end_reward_final3_dfg_pred"]
for exp_name in exp_names:
    print(exp_name)
    if exp_name == "end_reward_final2_dfg_pred":
        reward_type = exp_name.replace("_final2_dfg_pred","")
    else:
        reward_type = exp_name.replace("_final3_dfg_pred","")
    env_names = [ "traffic_ss"]
    threshs = [607.04]
    for i,env_name in enumerate(env_names):
        try:
            print(env_name)
            checkpoint_path2 = "PPO_preTrained/"+env_name+"/PPO_"+env_name+"_setting"+exp_name+".pth"
            # checking gv_to_gs on train set
            dname = "dataset/preprocessed/"+env_name+"_d2_"
            test_data = dname+'test_RL.pkl'
            td = pd.read_pickle(test_data)
            num_episodes = 10
            log_dir = "tests/"
            thresh = threshs[i]
            env = Custom_Environment(test_data, env_name = env_name, mode='verbose',mae=1.5, thresh=thresh,reward_type= reward_type)
            test_RL(env,num_episodes, checkpoint_path2,log_dir,env_name,exp_name,mode="testt")
        except Exception as e: print(e)


