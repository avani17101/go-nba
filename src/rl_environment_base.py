from RL_Agent.variable_action_space import IterableDiscrete
import gym
from gym import spaces
import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import pandas as pd
from utils import *

class Env_base(gym.Env):
    """
     Environment for Open AI Gym
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self,path, env_name, mode,mae,thresh,reward_type):
        super(Env_base, self).__init__()
        self.env_name = env_name
        # meta-data
        self.path = path
        self.batch_size = 1
        self.mode = mode
        self.episode = 0
        # desin mat, x,y (cur trace x, y)
        self.cur_prefix_len = 1
        self.reward_type = reward_type
        self.cur_case_idx = 0   #caseid_lis and idx -> caseid
        self.cur_trace_len = 0
        self.cur_trace_ind = 0
        self.x = None
        self.y = None
        self.reward_val = 0
        self.compliance = []
        self.first_act = None
        self.design_matrix = pd.read_pickle('dataset/preprocessed/'+self.env_name+'_design_mat.pkl')
        self.max_trace_len = get_trace_len(self.design_matrix)
        print(self.max_trace_len)
        # self.dic = {0: 1, 1: 1, 2: 0, 3: 1, 4: 0, 5: 0, 6: 1, 7: 0, 8: 1, 9: 0}

        # self.dic_acc = []
        # preprocs
        self.unique_event = [0] + sorted(self.design_matrix['class'].unique())
        print("self.unique_event",self.unique_event, len(self.unique_event))
        self.max_activities = len(self.unique_event)
        
        print("self.max_activities ",self.max_activities)
        self.selected_columns = np.arange(0,self.max_activities+1)
        
        self.initial_action_space = np.arange(0,self.max_activities)
        print("self.initial_action_space ",self.initial_action_space)
        self.design_matrix = pd.read_pickle(path)
        self.timestamp_loc = self.design_matrix.columns.get_loc('duration_time')
        self.caseId_lis = self.design_matrix["CaseID"].unique()
        print("num cases:",len(self.caseId_lis))

        # per prefix MAEs of KPI prediction model
        self.maes = get_maes(self.env_name, self.max_trace_len-1) 
        # model
        self.rnnG = torch.load("checkpoints/"+self.env_name+"/timestamp_prediction/prefix_1/rnnG.m")
        self.y_pred_softmax = None
        self.y_pred_event = None
        self.y_pred_timestamp = None
        self.y_truth_timestamp = None
        self.y_truth_event = None
        self.mae = mae
        self.cur_state = None
        self.thresh = thresh
        self.case_duration_preds = 0
        self.num_overall_goal_satisfied = 0
        self.overall_goal_satisfied = []
        self.overall_goal_not_satisfied = []
        self.lastk_act = {}
        self.num_overall_goal_not_satisfied = 0
        # self.per_case_error = []
        # self.case_mae = []
        self.accuracy_pred_per_event = {}
        self.accuracy_time_stamp = []
        self.accuracy_time_stamp_per_event = {}
        self.y_truth_list = []
        self.y_pred_list = []
        self.x_inn = []
        self.prev_action_one_hot = None
        self.y_truth_event_num = None
        self.y_pred_event = None
        self.chosen_action = None
        self.action_mask = np.ones(self.max_activities)
        self.y_truth_timestamp_list = []
        self.y_pred_timestamp_list = []
        self.gs_pred_cases = []
        self.gv_pred_cases = []
        # self.mae_final = 0
        self.percent_overall_gs = 0
        self.percent_overall_gv = 0
        self.percent_gv_which_became_gs, self.percent_gs_which_became_gv = 0,0
        self.accuracy_lastk = []
        self.accuracy_last1 = []
        self.accuracy_last2 = []
        self.accuracy_last3 = []
        self.deviation_from_goal_thresh_gs = [] 
        self.deviation_from_goal_thresh_gv = []

        self.getXY()
        first_activity_one_hot = self.x[:, 0, self.unique_event].detach().numpy()[0]
        first_act = np.where(first_activity_one_hot==1.)[0][0]
        self.possible_actions = get_available_actions(first_act,dset=self.env_name)
        self.obs = []
        self.action_space = IterableDiscrete(self.max_activities)
        low_array, high_array = get_obs_bounds2(self.design_matrix)
        self.observation_space = spaces.Box(low=low_array, high=high_array)
        print(self.observation_space)
        # self.prev_y_truth_event = first_act
        # self.prev_y_pred_event = first_act #for first activty of each trace- take y_pred = y_truth (since no y_pred data initially)
        
        self.percent_gs, self.gs_cases, self.gv_cases = self.find_percent_gs(self.design_matrix,self.thresh)
        self.change_action_space()
    
    def getXY(self):
        caseid = self.caseId_lis[self.cur_case_idx]
        gr = self.design_matrix[self.design_matrix['CaseID'] == caseid]
        self.cur_trace_len = len(list(gr[['CaseID']]))
        gr = gr.reset_index(drop=True)

        # adding a new row at the bottom of each case to denote the end of a case
        new_row = [0] * gr.shape[1]
        gr.loc[gr.shape[0]] = new_row
        gr.iloc[gr.shape[0] - 1, gr.columns.get_loc('0')] = 1  # End of line is denoted by class 0
        
        temp = torch.tensor(gr.values, dtype=torch.float, requires_grad=False)
        temp_shifted = torch.tensor(gr[['duration_time','class']].values, dtype=torch.float, requires_grad=False)

        self.x = pad_sequence([temp], batch_first=True)
        first_activity_one_hot = self.x[:, 0, self.unique_event].detach().numpy()[0]
        self.first_act = np.where(first_activity_one_hot==1.)[0][0]
        

        self.y = pad_sequence([temp_shifted], batch_first=True)
        self.y_truth_list = self.y[:,:,1][0].type(torch.int).numpy()
        self.cur_trace_len = int(self.x.shape[1])
        self.y_pred_list.append(self.first_act)
        # self.y_truth_list.append(self.first_act)

        # self.case_duration_preds += 
 

    def find_percent_gs(self,final_df, third_quartile):
        dat_group = final_df.groupby("CaseID")
        # total_iter = len(dat_group.ngroup())
        case_duration_dic = {}
        for name, gr in dat_group:
            case_duration_dic[name] = gr['duration_time'].sum()
        cases_gs = []
        cases_gv = []
        for k,v in case_duration_dic.items():
            if v <= third_quartile:
                cases_gs.append(k)
            else:
                cases_gv.append(k)   
        gs_percent = len(cases_gs)/(len(cases_gs)+len(cases_gv))
        return gs_percent, cases_gs, cases_gv
    
    def reward(self):
        pass

    def step(self, action):
        # Execute one time step within the environment
        self.action = action
        self.current_step += 1
        self.get_next_DLpred(action)
        reward_val = self.reward()
        self.cur_prefix_len += 1
        obs, reward, done = self.cur_state, reward_val, self.terminal()
        self.obs = obs
        if self.chosen_action == 0 and done == False:
            self.cur_case_idx += 1
            self.cur_prefix_len = 1
            self.x_inn = []
            self.case_duration_preds = 0
            self.getXY()
            self.possible_actions = get_available_actions(self.first_act,dset=self.env_name)
            self.rnnG = torch.load("checkpoints/"+self.env_name+"/timestamp_prediction/prefix_"+str(self.cur_prefix_len)+"/rnnG.m")
            self.change_action_space()
        return obs, reward, done, {}
    
    def separate_event_time(self,y_pred_last,actions):
        if self.y_truth_event != None:
            self.y_truth_event_num = int(self.y_truth_event[0][0][0])
        else:
            self.y_truth_event_num = None
    
        self.chosen_action = actions
        
        self.y_pred_list.append(actions)
        self.y_pred_timestamp = y_pred_last[:, :, 0].view((self.batch_size, 1, -1))
        self.case_duration_preds += self.y_pred_timestamp.detach().numpy()[0][0][0]  #take summation prefix 2, .. trace_len: prefix = 1 (first activity's) duration is always taken as zero(start) for each case 
        self.possible_actions = get_available_actions(actions,self.env_name)  #change action space for next activity according to current y_truth num
        self.change_action_space()

    def get_next_DLpred(self,action):
        self.create_state()
        y_pred2 = None
        # different vals sent to model according to gt available or not
        if self.cur_prefix_len < self.cur_trace_len:
            self.y_truth_timestamp = self.y[:, self.cur_prefix_len, 0].view(1, 1, -1).detach()
            self.y_truth_event = self.y[:, self.cur_prefix_len, 1].view(1, 1, -1).detach()
            # get prediction from DL model
            self.rnnG = torch.load("checkpoints/"+self.env_name+"/timestamp_prediction/prefix_"+str(self.cur_prefix_len)+"/rnnG.m")
            action_one_hot = convert_y_one_hot(torch.tensor(action),num_classes = self.max_activities)
            self.prev_action_one_hot = action_one_hot
            new_row = torch.cat((action_one_hot,torch.tensor([-1])),0).unsqueeze(0).unsqueeze(0)
            # print("cur_prefix_len", self.cur_prefix_len)
            # print(np.array(self.x).shape, np.array(self.x_inn).shape)
            # print(self.x_inn)

            if self.cur_prefix_len == 1:
                self.x_inn = torch.cat((self.x[:, :self.cur_prefix_len, self.selected_columns],new_row),dim=1).float()
            else:
                self.x_inn = torch.cat((self.x_inn[:, -self.cur_prefix_len:, self.selected_columns],new_row),dim=1).float()
                x_inn2 = torch.cat((self.x[:, -self.cur_prefix_len:, self.selected_columns],new_row),dim=1).float()
                y_pred2 = self.rnnG(x_inn2)

            # predict 
            y_pred = self.rnnG(self.x_inn)
            
            y_pred_last = y_pred[0: self.batch_size, self.cur_prefix_len - 1, :]
            y_pred_last = y_pred_last.view((1, 1, -1))
            # print("x_inn",self.x_inn)
            # print("comp: y_pred {}, y_pred2 {}".format(y_pred, y_pred2))

        else: 
            self.y_truth_timestamp = None  #gt vals not available for greater than trace_len prefixes
            self.y_truth_event = None  
            # taking higest trace len model
            self.rnnG = torch.load("checkpoints/"+self.env_name+"/timestamp_prediction/prefix_"+str(self.cur_trace_len)+"/rnnG.m")
            action_one_hot = convert_y_one_hot(torch.tensor(action),num_classes = self.max_activities)
            self.prev_action_one_hot = action_one_hot
            new_row = torch.cat((action_one_hot,torch.tensor([-1])),0).unsqueeze(0).unsqueeze(0)
            # taking last cur_trace_len vals for passing to dl model
            self.x_inn = torch.cat((self.x_inn[:, -self.cur_trace_len:, self.selected_columns],new_row),dim=1).float()
        
            # predict 
            y_pred = self.rnnG(self.x_inn)
            y_pred_last = y_pred[0: self.batch_size, self.cur_trace_len - 1, :]
            y_pred_last = y_pred_last.view((1, 1, -1))
        
        #trace_gt, action -> timestamp
        #trace_explored, action -> timestamp
        #mae*prefix_number
        #sum(mae_prefix)
        
        # separating y as y_ and y_timestamp
        self.separate_event_time(y_pred_last,action)
        self.y_pred_action = self.y_pred_softmax
        y_pred = self.y_pred_timestamp.squeeze(0).squeeze(0).detach()
        self.x_inn[:,-1,-1] = y_pred
        # print(self.x_inn)


    def close(self):
        super().close()

    # def goal_satisfied(self):
    #     if (self.y_pred_timestamp  - self.mae) <= self.y_truth_timestamp:
    #         return True
    #     return False
        
    def create_state(self):  # one_hot_prev_act, prev_time
        if len(self.x_inn):
            y_pred = self.y_pred_timestamp.squeeze(0).squeeze(0).detach()
            self.cur_state = torch.cat((self.prev_action_one_hot.detach() ,y_pred),0).float().numpy()
            # print(self.cur_state.shape)
            # print("cur state ifff",self.cur_state)
        else: #at beginning prefx len 1

            self.cur_state = self.x[:,self.cur_prefix_len-1, :-3][0].numpy()
        # print("cur state",self.cur_state)
        # print("cur_state", self.cur_state.shape)
             
            
    
    def convert_y_one_hot(y, num_classes):
        y = y.type(torch.int64)
        y_one_hot = F.one_hot(y, num_classes=num_classes)
        return y_one_hot.type(torch.double)
        
    def change_action_space(self):
        self.action_space.enable_actions(self.possible_actions)
        a_ = []
        for ac in self.initial_action_space:
            if ac not in self.possible_actions:
                a_.append(ac)
        self.action_space.disable_actions(a_)
        mask = np.zeros(self.max_activities)
        for i,ac in enumerate(self.initial_action_space):
            if ac in self.possible_actions:
                mask[i] = 1
        self.action_mask = mask
        ac_lis = []
        for ac in self.action_space:
            ac_lis.append(ac)
    
    def terminal(self):
        if self.cur_case_idx >= len(self.caseId_lis)-2:
            return True
        return False 
    
    def get_gs_gv_percent(self,gs_cases, gv_cases, gs_pred_cases, gv_pred_cases):
        cases_gs_gv = np.intersect1d(gs_cases, gv_pred_cases) 
        cases_gv_gs = np.intersect1d(gv_cases, gs_pred_cases) #gt was gv, but pred went gs
        percent_gv_which_became_gs = 0
        percent_gs_which_became_gv = 0
        if len(gv_cases)!=0:
            percent_gv_which_became_gs = len(cases_gv_gs)/len(gv_cases)
        if len(gs_cases)!=0:
            percent_gs_which_became_gv = len(cases_gs_gv)/len(gs_cases)

        return percent_gv_which_became_gs, percent_gs_which_became_gv
        
    def reset(self):
        # Reset the state of the environment to an initial state
        self.current_step = 0
        

           
        if self.episode !=0:
            self.overall_goal_satisfied.append(self.num_overall_goal_satisfied/len(self.caseId_lis))
            self.overall_goal_not_satisfied.append(self.num_overall_goal_not_satisfied/len(self.caseId_lis))
        
        self.percent_gv_which_became_gs, self.percent_gs_which_became_gv = self.get_gs_gv_percent(self.gs_cases, self.gv_cases, self.gs_pred_cases, self.gv_pred_cases)
        
        self.gs_pred_cases, self.gv_pred_cases = [],[]
        self.deviation_from_goal_thresh_gs = []
        self.deviation_from_goal_thresh_gv = []
        self.num_overall_goal_satisfied = 0
        self.num_overall_goal_not_satisfied = 0
        self.per_case_error = []
        self.y_truth_list = []
        self.y_pred_list = []
        self.num_overall_goal_satisfied = 0
        self.case_duration_preds = 0
        self.num_overall_goal_not_satisfied = 0
        self.prev_action_one_hot = None
        self.compliance = []
        # self.per_case_error = []
        # self.case_mae = []
        self.accuracy_pred_per_event = {}
        self.accuracy_time_stamp = []
        self.accuracy_time_stamp_per_event = {}
        self.accuracy_last_k = []
        self.accuracy_last1 = []
        self.accuracy_last2 = []
        self.accuracy_last3 = []
        self.x_inn = []
        self.lastk_act = {}
        self.cur_prefix_len = 1
        self.cur_case_idx = 0   #caseid_lis and idx -> caseid
        self.cur_trace_len = 0
        self.cur_trace_ind = 0
        self.getXY()
        first_activity_one_hot = self.x[:, 0, self.unique_event].detach().numpy()[0]
        first_act = np.where(first_activity_one_hot==1.)[0][0]
        self.possible_actions = get_available_actions(first_act,dset=self.env_name)
        self.change_action_space()
        # self.prev_y_truth_event = first_act
        # self.prev_y_pred_event = first_act #for first activty of each trace- take y_pred = y_truth (since no y_pred data initially)
       
        # model
        self.rnnG = torch.load("checkpoints/"+self.env_name+"/timestamp_prediction/prefix_1/rnnG.m")
        self.mae = 2
        self.create_state()
        self.episode += 1
        return self.cur_state

    def plotlastk(self, lastk):
        plt.figure()
        plt.bar(lastk.keys(), lastk.values())
        plt.savefig(self.env_name+self.reward_type+'lastkplot.png')

    def render(self, mode="human", close=False):
        
        # self.mae_final = np.array(self.per_case_error).mean()
       
        self.percent_overall_gs = self.num_overall_goal_satisfied/(self.num_overall_goal_satisfied+self.num_overall_goal_not_satisfied)
        self.percent_overall_gv = self.num_overall_goal_not_satisfied/(self.num_overall_goal_satisfied+self.num_overall_goal_not_satisfied)
        self.percent_gv_which_became_gs, self.percent_gs_which_became_gv = self.get_gs_gv_percent(self.gs_cases, self.gv_cases, self.gs_pred_cases, self.gv_pred_cases)
        self.compliance_per = np.mean(np.array(self.compliance))

        if 0 in self.lastk_act:
            del self.lastk_act[0]
        self.plotlastk(self.lastk_act)
        print("last k activities: ", self.lastk_act)
        print("dfg complaince: ",self.compliance_per)
        # print("per case mean absolute error of timestamp(in days): ", self.mae_final)
        print("The accuracy of the model on last k activities: ",np.mean(self.accuracy_lastk))
        print("The accuracy of the model on last activity: ",np.mean(self.accuracy_last1))
        print("The accuracy of the model on last 2 activities: ",np.mean(self.accuracy_last2))
        print("The accuracy of the model on last 3 activities: ",np.mean(self.accuracy_last3))
        print("percent overall goal satisfied in preds: ",self.percent_overall_gs)
        print("percent overall goal not satisfied in preds: ",self.percent_overall_gv)
        print("percent overall goal satisfied in gt: ",self.percent_gs)
        print("percent overall goal not satisfied in gt: ",1-self.percent_gs)
        print("percent_gv_which_became_gs: ",self.percent_gv_which_became_gs)
        print("percent_gs_which_became_gv: ",self.percent_gs_which_became_gv)
        print("deviation from goal threshold in gs(desired): ", np.mean(np.array(self.deviation_from_goal_thresh_gs)))
        print("deviation from goal threshold in gv: ", np.mean(np.array(self.deviation_from_goal_thresh_gv)))
    
