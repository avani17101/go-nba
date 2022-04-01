import torch
import numpy as np
import pandas as pd
from utils import *
from rl_environment_base2 import Env_base
from dfg import check_dfg_compliance

class Custom_Environment(Env_base):
    """
     Environment for Open AI Gym
    """
    def reward_k_act(self,k):
        delta = 0
        self.reward_val = 0
        if(self.y_pred_timestamp.detach().numpy()[0][0][0] < 0):
            self.y_pred_timestamp = torch.tensor([[[0.]]]) 

        if check_dfg_compliance(self.y_pred_list[-2] ,self.y_pred_list[-1], dset=self.env_name):
            self.compliance.append(1)
        else:
            print("prev chosen {} chosen {}".format(self.prev_action,self.chosen_action ))
            self.compliance.append(0)


        #END REWARD: reward val for end of process
        if self.chosen_action == 0: #end of trace
            # print("maes", sum(self.maes[:self.cur_prefix_len]))
            if self.case_duration_preds - sum(self.maes[:self.cur_prefix_len]) < self.thresh:  #goal satisfied
                self.num_overall_goal_satisfied += 1
                self.gs_pred_cases.append(self.caseId_lis[self.cur_case_idx])
                self.reward_val += 2*(self.thresh - self.case_duration_preds)/self.thresh
                self.deviation_from_goal_thresh_gs.append(abs(self.case_duration_preds-self.thresh))
            else:
                self.num_overall_goal_not_satisfied += 1
                self.gv_pred_cases.append(self.caseId_lis[self.cur_case_idx])
                self.reward_val += 2*(self.thresh - self.case_duration_preds)/self.thresh
                self.deviation_from_goal_thresh_gv.append(self.case_duration_preds-self.thresh)

            self.case_duration_preds = 0
            # self.per_case_error.append(np.array(self.case_mae).mean())
            # self.case_mae = []

            # print("self.y_pred_list[-1:]", self.y_pred_list[-1:])  #excluding balancing reward for prediction 0: end of trace
            # BALANCING REWARD: last k events rewards
            # last k activity reward (last activity is 0 always, hence excluding end of trace(0) we take last 3rd and last 2nd activity)
            
            
            lenn = min(k+2,len(self.y_pred_list))
            if lenn<=2:
                lenn = 3
            for i in range(2,lenn):
                if(self.y_pred_list[-i:][0] == self.y_truth_list[-i:][0]): #action chosen by RL agent = gt
                    self.reward_val += 0.5
                    self.accuracy_lastk.append(1)
                else:
                    self.reward_val -= 0.5
                    self.accuracy_lastk.append(0)

                if self.y_pred_list[-i:][0] in self.lastk_act:
                    self.lastk_act[self.y_pred_list[-i:][0]] += 1
                else:
                    self.lastk_act[self.y_pred_list[-i:][0]] = 1

            lenn = min(5,len(self.y_pred_list))
            for i in range(2,lenn):  
                if i == 1+1:
                    if(self.y_pred_list[-i:][0] == self.y_truth_list[-i:][0]): #action chosen by RL agent = gt
                            self.accuracy_last1.append(1)
                            self.accuracy_last2.append(1)
                            self.accuracy_last3.append(1)
                    else:
                        self.accuracy_last1.append(0)
                        self.accuracy_last2.append(0)
                        self.accuracy_last3.append(0)
                if i == 2+1:
                    if(self.y_pred_list[-i:][0] == self.y_truth_list[-i:][0]): #action chosen by RL agent = gt
                            self.accuracy_last2.append(1)
                            self.accuracy_last3.append(1)
                    else:
                        self.accuracy_last2.append(0)
                        self.accuracy_last3.append(0)
                if i == 3+1:
                    if(self.y_pred_list[-i:][0] == self.y_truth_list[-i:][0]): #action chosen by RL agent = gt
                            self.accuracy_last3.append(1)
                    else:
                        self.accuracy_last3.append(0)

                
            self.y_pred_list = []
            self.y_pred_list = []

    def reward(self):
        if self.reward_type == "last_1_act_and_end_reward":
            self.reward_k_act(k=1)
            return self.reward_val

        if self.reward_type == "last_2_act_and_end_reward":
            self.reward_k_act(k=2)
            return self.reward_val

        if self.reward_type == "last_3_act_and_end_reward":
            self.reward_k_act(k=3)
            return self.reward_val

        if self.reward_type == "end_reward":
            delta = 0
            self.reward_val = 0
            if(self.y_pred_timestamp.detach().numpy()[0][0][0] < 0):
                self.y_pred_timestamp = torch.tensor([[[0.]]])

            if check_dfg_compliance(self.y_pred_list[-2] ,self.y_pred_list[-1], dset=self.env_name):
                self.compliance.append(1)
            else:
                print("prev chosen {} chosen {}".format(self.prev_action,self.chosen_action ))
                self.compliance.append(0)
            # reward val for end of process
            if self.chosen_action == 0: #end of trace
                # print(self.y.shape[1]-1)
                if self.case_duration_preds - sum(self.maes[:self.cur_prefix_len]) < self.thresh:  #goal satisfied
                    self.num_overall_goal_satisfied += 1
                    self.gs_pred_cases.append(self.caseId_lis[self.cur_case_idx])
                    self.reward_val += 2*(self.thresh - self.case_duration_preds)/self.thresh
                    self.deviation_from_goal_thresh_gs.append(abs(self.case_duration_preds-self.thresh))
                else:
                    self.num_overall_goal_not_satisfied += 1
                    self.gv_pred_cases.append(self.caseId_lis[self.cur_case_idx])
                    self.reward_val += 2*(self.thresh - self.case_duration_preds)/self.thresh
                    self.deviation_from_goal_thresh_gv.append(self.case_duration_preds-self.thresh)

                self.case_duration_preds = 0
                lenn = min(5,len(self.y_pred_list))
                for i in range(2,lenn):  
                    if self.y_pred_list[-i:][0] in self.lastk_act:
                        self.lastk_act[self.y_pred_list[-i:][0]] += 1
                    else:
                        self.lastk_act[self.y_pred_list[-i:][0]] = 1 
                    if i == 1+1:
                        if(self.y_pred_list[-i:][0] == self.y_truth_list[-i:][0]): #action chosen by RL agent = gt
                                self.accuracy_last1.append(1)
                                self.accuracy_last2.append(1)
                                self.accuracy_last3.append(1)
                        else:
                            self.accuracy_last1.append(0)
                            self.accuracy_last2.append(0)
                            self.accuracy_last3.append(0)
                    if i == 2+1:
                        if(self.y_pred_list[-i:][0] == self.y_truth_list[-i:][0]): #action chosen by RL agent = gt
                                self.accuracy_last2.append(1)
                                self.accuracy_last3.append(1)
                        else:
                            self.accuracy_last2.append(0)
                            self.accuracy_last3.append(0)
                    if i == 3+1:
                        if(self.y_pred_list[-i:][0] == self.y_truth_list[-i:][0]): #action chosen by RL agent = gt
                                self.accuracy_last3.append(1)
                        else:
                            self.accuracy_last3.append(0)

                    
                self.y_pred_list = []
                self.y_pred_list = []
            return self.reward_val
            

    
    
    