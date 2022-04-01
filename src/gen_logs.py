import pandas as pd
from rl_environment import  Custom_Environment
from  RL_Agent.test_RL_Agent import test_RL
import numpy as np
import csv 

def gencsv():
    codes = ['z','zz','zzz']
    for code in codes:
        csvfile = open(code+".csv", 'w+')
        # creating a csv writer object 
        csvwriter = csv.writer(csvfile) 
        rows = []
        exp_names = ["last_3_act_and_end_reward_final"+code+"_dfg_pred_best_model","last_2_act_and_end_reward_final"+code+"_dfg_pred_best_model","last_1_act_and_end_reward_final"+code+"_dfg_pred_best_model","end_reward_final"+"z"+"_dfg_pred_best_model"]
        exp_names.reverse()
        name = ["r","r'k=1","r'k=2","r'k=3"]
        mode = 'test'
        # writing the fields 
        env_names = ["helpdesk", "bpi_12_w", "traffic_ss"]
	    log_dir = "tests_avg/"
        for env_name in env_names:
            for i,exp_name in enumerate(exp_names):
                log_path = log_dir+"PPO_{}_setting{}_{}.csv".format(env_name,exp_name,mode)
                df = pd.read_csv(log_path)
                rows.append([env_name, name[i]]+list(df.mean().values))
                fields = ["Dataset","Exp"]+list(df.mean().index)
        # writing the data rows 
        csvwriter.writerow(fields) 
        csvwriter.writerows(rows) 
        csvfile.close()
        df = pd.read_csv(code+".csv")
        print(df)


def testing(exp_names):
    reward_types = ["last_3_act_and_end_reward","last_2_act_and_end_reward","last_1_act_and_end_reward", "end_reward"]
    for j,exp_name in enumerate(exp_names):
        reward_type = reward_types[j]
        print(exp_name)
        print(reward_type)
        env_names = ["helpdesk", "bpi_12_w", "traffic_ss", "bpi2019"]
        threshs = [13.89, 24.001, 607.04, 94.015]
        for i,env_name in enumerate(env_names):
            try:
                print(env_name)
                checkpoint_path2 = "PPO_preTrained/"+env_name+"/PPO_"+env_name+"_setting"+exp_name+".pth"
                # checking gv_to_gs on train set

                dname = "dataset/preprocessed/"+env_name+"_d2_"
                test_data = dname+'test_RL.pkl'
                td = pd.read_pickle(test_data)
                num_episodes = 10
                log_dir = "tests_avg/"
                thresh = threshs[i]
                env = Custom_Environment(test_data, env_name = env_name, mode='verbose',mae=1.5, thresh=thresh,reward_type= reward_type)
                test_RL(env,num_episodes, checkpoint_path2,log_dir,env_name,exp_name,mode="testt")
            except Exception as e: print(e)

# exp_names = ["last_3_act_and_end_reward_finalz_dfg_pred_best_model","last_2_act_and_end_reward_finalz_dfg_pred_best_model","last_1_act_and_end_reward_finalz_dfg_pred_best_model","end_reward_finalz_dfg_pred_best_model"]
# testing(exp_names)
# exp_names = ["last_3_act_and_end_reward_finalzz_dfg_pred_best_model","last_2_act_and_end_reward_finalzz_dfg_pred_best_model","last_1_act_and_end_reward_finalzz_dfg_pred_best_model"]
# testing(exp_names)
# exp_names = ["last_3_act_and_end_reward_finalzzz_dfg_pred_best_model","last_2_act_and_end_reward_finalzzz_dfg_pred_best_model","last_1_act_and_end_reward_finalzzz_dfg_pred_best_model"]
# testing(exp_names)

gencsv()



