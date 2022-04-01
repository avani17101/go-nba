import csv
from datetime import datetime
from RL_Agent.maskable_PPO import *
import numpy as np
from tqdm import tqdm

np.set_printoptions(linewidth=1000)

def train_RL(env, episodes, checkpoint_dir, log_dir, env_name, exp_name,  use_action_mask, checkpoint_path, resume_training):
    """
    Training the RL agent 
    args: 
        env: the environment object
        episodes: number of episodes to test upon
        checkpoint_dir: the path where checkpoint to be saved
        log_dir: the logs path for trained model
        env_name: name of environment (to be used for loading and saving logs)
        exp_name: experiment name (to be used for loading and saving logs)
    returns:
        checkpoint_path2: path to best model's checkpoint (best model is model in episode with highest reward)
        log_path: the path to saved logs of training
    """
    has_continuous_action_space = False
    max_ep_len = 1000000000                # max timesteps in one episode
    max_training_timesteps = int(max_ep_len)   # break training loop if timeteps > max_training_timesteps
    print_freq = max_ep_len * 4     # print avg reward in the interval (in num timesteps)
    log_freq = max_ep_len * 2       # log avg reward in the interval (in num timesteps)
    save_model_freq = int(2e4)      # save model frequency (in num timesteps)
    action_std = None



    ################ PPO hyperparameters ################


    update_timestep = max_ep_len * 4      # update policy every n timesteps
    K_epochs = 40               # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    gamma = 0.99                # discount factor

    lr_actor = 0.0003       # learning rate for actor network
    lr_critic = 0.001       # learning rate for critic network


    #####################################################

    # state space dimension
    state_dim = env.observation_space.shape[0]

    # action space dimension
    if has_continuous_action_space:
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n

    #####################################################


    ############# print all hyperparameters #############
    print("max timesteps per episode : ", max_ep_len)
    print("model saving frequency : " + str(save_model_freq) + " timesteps")
    print("log frequency : " + str(log_freq) + " timesteps")
    print("printing average reward over episodes in last : " + str(print_freq) + " timesteps")
    print("--------------------------------------------------------------------------------------------")
    print("state space dimension : ", state_dim)
    print("action space dimension : ", action_dim)
    print("--------------------------------------------------------------------------------------------")
    print("Initializing a discrete action space policy")
    print("--------------------------------------------------------------------------------------------")
    print("PPO update frequency : " + str(update_timestep) + " timesteps") 
    print("PPO K epochs : ", K_epochs)
    print("PPO epsilon clip : ", eps_clip)
    print("discount factor (gamma) : ", gamma)
    print("--------------------------------------------------------------------------------------------")
    print("optimizer learning rate actor : ", lr_actor)
    print("optimizer learning rate critic : ", lr_critic)



    #####################################################

    print("============================================================================================")

    ################# training procedure ################

    # initialize a PPO agent
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)

    if resume_training:
        # preTrained weights directory
        print("loading network from : " + checkpoint_path)
        ppo_agent.load(checkpoint_path)

    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)

    print("============================================================================================")

    # printing and logging variables
    time_step = 0
    i_episode = 1
    reward_lis = []
    fields = ["episode","reward","accuracy_last","accuracy_last2","accuracy_last3","gs","gv","gv_turned_gs","gs_turned_gv"]
    rows = []
    count = 0
    # defining log, checkpoint paths
    log_path = log_dir+"PPO_{}_setting{}.csv".format(env_name,exp_name)
    checkpoint_path = checkpoint_dir + "PPO_{}_setting{}.pth".format(env_name,exp_name)
    print("save checkpoint path : " + checkpoint_path)
    f = open(log_dir+"best_model_espisode.txt","w+")
    checkpoint_path2 = checkpoint_dir  + "PPO_{}_setting{}_best_model.pth".format(env_name,exp_name)

    csvfile = open(log_path, 'w') 
    # creating a csv writer object
    csvwriter = csv.writer(csvfile)
    # writing the fields
    csvwriter.writerow(fields)

    # training loop
    max_episodes = episodes
    max_avg_reward = 0
    for  i_episode in range(1,max_episodes+1):    
        state = env.reset()
        current_ep_reward = 0
        avg_reward = 0
        prev_avg_reward = 0
        episode_reward = []
        for t in range(1, max_ep_len+1):
            # select action with policy
            action = 0
            if use_action_mask == True:
                action = ppo_agent.select_action(state, env.action_mask)
            else:
                action = ppo_agent.select_action(state, [])
            state, reward, done, _ = env.step(action)
            episode_reward.append(reward)
            #print(reward)
            # saving reward and is_terminals
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(done)
            
            time_step +=1
            current_ep_reward += reward

            # update PPO agent
            if time_step % update_timestep == 0:
                if use_action_mask == True:
                    ppo_agent.update(env.action_mask)
                else:
                    ppo_agent.update(None)
                    
            if done:
                break
                
        episode_reward = np.array(episode_reward)
        prev_avg_reward = avg_reward
        avg_reward = episode_reward.mean()
        reward_lis.append(avg_reward)
        
    
        print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step, avg_reward))
        env.render()
        rows.append([i_episode,avg_reward, np.mean(env.accuracy_last1), np.mean(env.accuracy_last2), np.mean(env.accuracy_last3), env.percent_overall_gs, env.percent_overall_gv, env.percent_gv_which_became_gs,env.percent_gs_which_became_gv])

        if avg_reward - prev_avg_reward < 1e8:
            count += 1
            if count == 10:
                print("terminating at episode {}",i_episode)
                if avg_reward > prev_avg_reward:
                    print("--------------------------------------------------------------------------------------------")
                    print("saving model at : " + checkpoint_path)
                    ppo_agent.save(checkpoint_path)
                    print("model saved")
                break
            else:
                count = 0
    
        if avg_reward > max_avg_reward: #best model
            max_avg_reward = avg_reward
            print("--------------------------------------------------------------------------------------------")
            f.write(str(i_episode))
            print("saving best model at : " + checkpoint_path2)
            ppo_agent.save(checkpoint_path2)
            print("best model saved")

        if i_episode == 10:
            print("--------------------------------------------------------------------------------------------")
            print("saving model at : " + checkpoint_path.split('.')[0]+"epi10.pth")
            ppo_agent.save(checkpoint_path)
            print("model saved")

        if i_episode == 100:
            print("--------------------------------------------------------------------------------------------")
            print("saving model at : " + checkpoint_path.split('.')[0]+"epi100.pth")
            ppo_agent.save(checkpoint_path)
            print("model saved")

        if i_episode == 300:
            print("--------------------------------------------------------------------------------------------")
            print("saving model at : " + checkpoint_path.split('.')[0]+"epi300.pth")
            ppo_agent.save(checkpoint_path)
            print("model saved")
               
        if i_episode %10 == 0:
            print("--------------------------------------------------------------------------------------------")
            print("saving model at : " + checkpoint_path)
            ppo_agent.save(checkpoint_path)
            print("model saved")
            
    # writing the data rows
    csvwriter.writerows(rows)
    csvfile.close()

    # save model weights
    print("--------------------------------------------------------------------                                                                                                                                                                                                                                                                                                                                                                                                             ------------------------")
    print("saving model at : " + checkpoint_path)
    ppo_agent.save(checkpoint_path)
    print("model saved")
    print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
    print("--------------------------------------------------------------------------------------------")
    
    f.close()
    # print total training time
    print("============================================================================================")
    end_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print("============================================================================================")
    print("============================================================================================")
    return checkpoint_path, log_path



