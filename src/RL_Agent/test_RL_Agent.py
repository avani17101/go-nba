import csv
from RL_Agent.maskable_PPO import *
import numpy as np
np.set_printoptions(linewidth=1000)

def test_RL(env,episodes,checkpoint_path, log_dir,env_name,exp_name, mode):
    """
    Testing the RL agent 
    args: 
        env: the environment object
        episodes: number of episodes to test upon
        checkpoint_path: the path from where checkpoint to be loaded
        log_dir: the logs path for trained model
        env_name: name of environment (to be used for loading and saving logs)
        exp_name: experiment name (to be used for loading and saving logs)
        mode: test/val (uses to save log)
    """
    ################## hyperparameters ##################
    has_continuous_action_space = False

    max_ep_len = 8000000
    action_std = None

    total_test_episodes = episodes # total num of testing episodes

    K_epochs = 80               # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    gamma = 0.99                # discount factor

    lr_actor = 0.0003           # learning rate for actor
    lr_critic = 0.001           # learning rate for critic

    #####################################################
    # state space dimension
    state_dim = env.observation_space.shape[0]

    # action space dimension
    if has_continuous_action_space:
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n

    # initialize a PPO agent
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)

    # preTrained weights directory
    print("loading network from : " + checkpoint_path)
    ppo_agent.load(checkpoint_path)
    fields = ["episode","reward","accuracy_last","accuracy_last2","accuracy_last3","gs","gv","gv_turned_gs","gs_turned_gv","lastk_act"]
    rows = []
    # log results path
    log_path = log_dir+"PPO_{}_setting{}_{}.csv".format(env_name,exp_name,mode)
    csvfile = open(log_path, 'w') 
    # creating a csv writer object
    csvwriter = csv.writer(csvfile)
    # writing the fields
    csvwriter.writerow(fields)

    print("--------------------------------------------------------------------------------------------")
    test_running_reward = 0
    for ep in range(1, total_test_episodes+1):
        ep_reward = []
        state = env.reset()
        for t in range(1, max_ep_len+1):
            action = ppo_agent.select_action(state,env.action_mask)
            state, reward, done, _ = env.step(action)
            ep_reward.append(reward)
            if done:
                break
        # clear buffer    
        ppo_agent.buffer.clear()
        mean_ep_reward =  np.array(ep_reward).mean()
        test_running_reward  += mean_ep_reward
        print('Episode: {} \t\t Reward: {}'.format(ep,  mean_ep_reward))
        env.render()
        rows.append([ep, mean_ep_reward, np.mean(env.accuracy_last1), np.mean(env.accuracy_last2), np.mean(env.accuracy_last3), env.percent_overall_gs, env.percent_overall_gv, env.percent_gv_which_became_gs,env.percent_gs_which_became_gv,env.lastk_act])

    env.close()
    csvwriter.writerows(rows)
    csvfile.close()


    print("============================================================================================")
    avg_test_reward = test_running_reward / total_test_episodes
    avg_test_reward = round(avg_test_reward, 2)
    print("average test reward : " + str(avg_test_reward))
    print("============================================================================================")

    import pandas as pd
    df = pd.read_csv(log_path)
    
    with open("res.txt","a") as f:
        print(env_name,file=f)
        print(exp_name, file=f)
        print(df.mean(),file=f)



