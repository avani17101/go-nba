
from utils import calc_third_quartile, get_unique_act, get_compliant_cases 
import numpy as np
import os
import pickle
import pandas as pd
import random
from statistics import mean, median
import utils

dataset = "traffic_ss"
df2 = pd.read_pickle('dataset/preprocessed/'+dataset+'_design_mat.pkl')

from dfg import get_dfg_graph
graph = get_dfg_graph(dataset)

graph[-1] = graph.pop(0)

# Python program to print all paths from a source to destination.
  
from collections import defaultdict
  
# This class represents a directed graph
# using adjacency list representation
class Graph:
  
    def __init__(self, vertices):
        # No. of vertices
        self.V = vertices
         
        # default dictionary to store graph
        self.graph = defaultdict(list)
  
    # function to add an edge to graph
    def addEdge(self, u, v):
        self.graph[u].append(v)
  
    '''A recursive function to print all paths from 'u' to 'd'.
    visited[] keeps track of vertices in current path.
    path[] stores actual vertices and path_index is current
    index in path[]'''
    def printAllPathsUtil(self, u, d, visited, path, paths):
         
        # Mark the current node as visited and store in path
        visited[u]= True
        path.append(u)
 
        # If current vertex is same as destination, then print
        # current path[]
        if u == d:
            paths.append(path.copy())
#             print(path)
        else:
            # If current vertex is not destination
            # Recur for all the vertices adjacent to this vertex
            for i in self.graph[u]:
                if visited[i]== False:
                    self.printAllPathsUtil(i, d, visited, path, paths)
                     
        # Remove current vertex from path[] and mark it as unvisited
        path.pop()
        visited[u]= False
  
  
    # Prints all paths from 's' to 'd'
    def printAllPaths(self, s, d):
 
        # Mark all the vertices as not visited
        visited =[False]*(self.V)
 
        # Create an array to store paths
        path = []
        paths = []
 
        # Call the recursive helper function to print all paths
        self.printAllPathsUtil(s, d, visited, path, paths)
        print("inside")
        print(paths)
        return paths
  

import numpy as np
def convert_y_one_hot(y, num_classes):
    y = y.type(torch.int64)
    y_one_hot = F.one_hot(y, num_classes=num_classes)
    return y_one_hot.type(torch.float64)


def get_gs_paths():
    '''
    gives gs paths (as prediction by KPI prediction model) from dfg graph 
    '''
    batch_size = 1
    selected_columns = np.arange(0,11)
    gs = 0
    gv = 0
    thresh = 13.89
    dic = {}
    gs_lis = []
    for act in range(max_activities):
        dic[act] = 0
    for path in paths:
        act = 0
        cur_prefix_len = 1
        action_one_hot = convert_y_one_hot(torch.tensor(act),num_classes = max_activities)
        new_row = torch.cat((action_one_hot,torch.tensor([0])),0).unsqueeze(0).unsqueeze(0)
        x_inn = new_row
        case_duration_preds = 0
        for i in range(1,len(path)):
            act = path[i]
            action_one_hot = convert_y_one_hot(torch.tensor(act),num_classes = max_activities)
            new_row = torch.cat((action_one_hot,torch.tensor([-1])),0).unsqueeze(0).unsqueeze(0)
            x_inn = torch.cat((x_inn[:, :cur_prefix_len, selected_columns],new_row),dim=1)
            rnnG = torch.load("checkpoints/"+env_name+"/timestamp_prediction/prefix_"+str(cur_prefix_len)+"/rnnG.m")
            y_pred = rnnG(x_inn.float())
            y_pred_last = y_pred[0: batch_size, cur_prefix_len - 1, :]
            y_pred_last = y_pred_last.view((1, 1, -1))
            y_pred_timestamp = y_pred_last[:, :, 0].view((batch_size, 1, -1))
            cur_prefix_len += 1
            case_duration_preds += y_pred_timestamp.detach().numpy()[0][0][0]  #take summation prefix 2, .. trace_len: prefix = 1 (first activity's) duration is always taken as zero(start) for each case 

        if case_duration_preds < thresh:
            gs += 1
            gs_lis.append(path)
            print("gs", path)

            for a in path:
                if a == -1:
                    a = 0
                dic[a] = 1

        else:
            gv += 1
            print("gv", path)


max_activities = len(graph)


g = Graph(len(graph))
for gg in graph:
    for dest in graph[gg]:
        g.addEdge(gg,dest)


s = -1
d = 0
print ("Following are all different paths from % d to % d :" %(s, d))

paths = g.printAllPaths(s, d)
print(paths)

import numpy as np
import torch
import torch.nn.functional as F

def convert_y_one_hot(y, num_classes):
    y = y.type(torch.int64)
    y_one_hot = F.one_hot(y, num_classes=num_classes)
    return y_one_hot.type(torch.float64)


def get_gs_paths():
    '''
    gives gs paths (as prediction by KPI prediction model) from dfg graph 
    '''
    batch_size = 1
    selected_columns = np.arange(0, max_activities+1)
    print(selected_columns)
    gs = 0
    gv = 0
    thresh = 13.89 + 0.971   # for helpdesk
    if env_name == "bpi_12_w":
        thresh = 24.001 + 0.499
    if env_name == "traffic_ss":
        thresh = 607.04 + 39.824
    if env_name == "bpi2019":
        thresh = 94.015 + 3.379
    else:
        thresh = utils.get_third_quartile(env_name)

    dic = {}
    gs_lis = []
    
    for act in range(max_activities):
        dic[act] = 0
    
    for path in paths:
        act = 0
        cur_prefix_len = 1
        action_one_hot = convert_y_one_hot(torch.tensor(act), num_classes = max_activities)
        new_row = torch.cat((action_one_hot,torch.tensor([0])), 0).unsqueeze(0).unsqueeze(0)
        x_inn = new_row
        case_duration_preds = 0
        for i in range(1,len(path)):
            act = path[i]
            action_one_hot = convert_y_one_hot(torch.tensor(act),num_classes = max_activities)
            new_row = torch.cat((action_one_hot,torch.tensor([-1])),0).unsqueeze(0).unsqueeze(0)
            x_inn = torch.cat((x_inn[:, :cur_prefix_len, selected_columns],new_row),dim=1)
            rnnG = torch.load("checkpoints/"+env_name+"/timestamp_prediction/prefix_"+str(cur_prefix_len)+"/rnnG.m")
            y_pred = rnnG(x_inn.float())
            y_pred_last = y_pred[0: batch_size, cur_prefix_len - 1, :]
            y_pred_last = y_pred_last.view((1, 1, -1))
            y_pred_timestamp = y_pred_last[:, :, 0].view((batch_size, 1, -1))
            cur_prefix_len += 1
            case_duration_preds += y_pred_timestamp.detach().numpy()[0][0][0]  #take summation prefix 2, .. trace_len: prefix = 1 (first activity's) duration is always taken as zero(start) for each case 

        if case_duration_preds < thresh:
            gs += 1
            gs_lis.append(path[1:-1])
           

            for a in path:
                if a == -1:
                    a = 0
                dic[a] = 1

        else:
            gv += 1
            
    return gs_lis


# -1 is start node, 0 is end
env_name = dataset
gs_lis = get_gs_paths()

print("gs statisfying paths according to kpi pred model")
print(gs_lis)


name = dataset+'_d2'
df2 = pd.read_pickle('dataset/preprocessed/'+name+"_test_RL.pkl")


cases_sat = []
dat_group = df2.groupby("CaseID")
for name, gr in dat_group:
    act = list(gr['class'])
    
    if act in gs_lis:
        cases_sat.append(name)



print("precent gs satisfying", len(cases_sat)/len(df2['CaseID'].unique()))



data_filtered = df2.loc[df2['CaseID'].isin(cases_sat)]

# saving model
name = dataset+'_d2'
pickle.dump(data_filtered, open(name+"_test_RL_filtered.pkl", "wb"))
