from dfg import checkCandidate, get_dfg_graph
import matplotlib.pyplot as plt
import numpy as np
import os                                                                                                                                                                                                                               
import pandas as pd
from statistics import mean, median
import torch.nn.functional as F

def get_available_actions(node,dset):
    """
    get next available actions according to dfg flow
    """
    graph = get_dfg_graph(dset)
    available_nodes = list(graph[node].keys())
    return available_nodes

def convert_y_one_hot(y,num_classes):
    """
    convert the y to num_classes sized one-hot vector
    """
    y_one_hot = F.one_hot(y, num_classes)
    return y_one_hot

def get_trace_len(df):
    """
    get the length of trace
    """
    cid = list(df['CaseID'])
    dic = {}
    for i in cid:
        if i in dic:
            dic[i] += 1
        else:
            dic[i] = 1
    trace_len = int(max(dic.values()))
    return trace_len

def num_occurance_atob(a,b,df):
    """
    find num occurance of activity a followed by activity b in dataframe df
    args: a: activity number (dtype: int)
          b: activity number (dtype: int)
          df: dataframe (pandas) where a, b occur (must have ActivityID column)
    returns: num_occurance   (dtype: int)
    """
    h = df[df["ActivityID"]==a]
    ind = list(h.index)
    oc = 0
    for i in ind:
        if i < 13709:
            next_act = int(df.iloc[i+1].ActivityID)
            if next_act == b:
                oc += 1
    return oc

def calc_third_quartile(lis):
    """
    calculate third quartile value for a list
    """
    lis.sort()
    size = len(lis)
    lis_upper_half = lis[size//2:-1]
    third_quartile = median(lis_upper_half)
    return third_quartile

def get_third_quartile(env_name):
    df2 = pd.read_pickle('dataset/preprocessed/'+env_name+'_design_mat.pkl')
    df = get_compliant_cases(df2,env_name)   
    del df2
    dat_group = df.groupby("CaseID")
    case_duration_dic = {}
    for name, gr in dat_group:
        case_duration_dic[name] = gr['duration_time'].sum()
    case_durations = list(case_duration_dic.values())
    third_quartile  = calc_third_quartile(case_durations)
    return third_quartile



def get_unique_act(df):
    """
    give unique activities from dataframe
    """
    unique_act = [0] + sorted(df['class'].unique())
    return unique_act

def get_compliant_cases(df2,dset):
    """
    get the dfg process flow compliant cases
    """
    final_df = []
    dat_group = df2.groupby('CaseID')
    for case, g in dat_group:
        seq = list(g['class'])
        if checkCandidate(seq,dset) and len(seq)>1:
            final_df.append(g)
        if len(seq)<1:
            print("prefix len 1")
    final_df = pd.concat(final_df)       
    return final_df


def plot_case_occ(df):
    occurances = []
    print("activity   occurance")
    for i in range(1,10):
        occ = len(df[df['class']==i])
        occurances.append(occ)
        print(str(i)+"             "+str(occ))

    activities = np.arange(1,10)
    plt.bar(activities, occurances)
    plt.xlabel("activity")
    plt.ylabel("occurance")

def plot(reward_lis):
    episodes = np.arange(0,len(reward_lis))
    plt.figure()
    plt.plot(episodes, reward_lis)
    plt.title("reward vs episodes, mean reward : "+str(mean(reward_lis)))
    plt.xlabel("episodes")
    plt.ylabel("reward")
  
def get_obs_bounds(df2):
    low_array = np.zeros(len(df2.columns))
    low_array[-2] = min(df2['class'])
    low_array[-1] = min(df2['CaseID'])
    high_array = np.ones(len(df2.columns))
    high_array[-4] = max(df2['duration_time'])
    high_array[-3] = max(df2['remaining_time'])
    high_array[-2] = max(df2['class'])
    high_array[-1] = max(df2['CaseID'])
    return low_array, high_array

def save_plot(log_path):
    """
    save episode vs reward and accuracy path given csv logs file path
    """
    log_df = pd.read_csv(log_path)
    ax = log_df.plot(x='episode',y=['reward','accuracy']) 
    graph_path = log_path.replace(".csv",".png") 
    ax.figure.savefig(graph_path)    
   
def checkpoint_log_paths(env_name):
    """
    create the checkpoint and log dirs and return them
    """
    log_dir = "PPO_logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_dir = log_dir + '/' + env_name + '/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    checkpoint_dir = "PPO_preTrained"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    checkpoint_dir = checkpoint_dir + '/' + env_name + '/'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # log_path = log_dir+"PPO_{}_setting{}.csv".format(env_name,exp_name)
    # checkpoint_path = directory + "PPO_{}_setting{}.pth".format(env_name,exp_name)
    # print("save checkpoint path : " + checkpoint_path)
    # checkpoint_path2 = directory + "PPO_{}_setting{}_best_model.pth".format(env_name,exp_name)
    return  checkpoint_dir, log_dir
