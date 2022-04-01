import numpy as np
import pandas as pd
import torch                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from dfg import check_dfg_compliance

def model_eval_test(dataset_name, csvwriter=None):
    '''
       This module is for validation and testing the Generator
       @param modelG: Generator neural network
       @param mode: 'validation', 'test', 'test-validation'
       @param obj: A data object created from "Input" class that contains the required information
       @return: The accuracy of the Generator
       '''
    # set the evaluation mode (this mode is necessary if you train with batch, since in test the size of batch is different)
    
    batch = 1
    #events = list(np.arange(0, len(obj.unique_event) + 1))
    predicted = []
    accuracy = []
    mae = []
    dfg_compliance_pred = []
    dfg_compliance_gt = []
    y_truth_list = []
    y_pred_last_event_list = []
    df = pd.read_pickle("dataset/preprocessed/"+dataset_name+"_design_mat.pkl")
    unique_event = [0] + sorted(df['class'].unique())
    events = list(np.arange(0, len(unique_event)))
    print("events",events)
    max_activities = len(unique_event)
    dur = 0
    dur_gt = 0
    num_overall_goal_satisfied = 0
    num_overall_goal_not_satisfied = 0
    num_overall_goal_satisfied_gt = 0
    num_overall_goal_not_satisfied_gt = 0
    row = []
    selected_columns = np.arange(0,max_activities+1)
    y_pred_last_event_list_prev = None
    group = df.groupby('CaseID')
    cur_prefix_len = 1
    prev_event = None
    max_prefix_len = 13
    thresh = 13.89
    if dataset_name == "helpdesk":
        thresh = 13.89 
        max_prefix_len = 13
    if dataset_name == "bpi_12_w":
        thresh = 18.28
        max_prefix_len = 73
    if dataset_name == "traffic_ss":
        thresh = 607.04
        max_prefix_len = 16

    for name,gr in group:
        gr = gr.reset_index(drop=True)
        new_row = [0] * gr.shape[1]
        gr.loc[gr.shape[0]] = new_row
        gr.iloc[gr.shape[0] - 1, gr.columns.get_loc('0')] = 1  # End of line is denoted by class 0
        temp = torch.tensor(gr.values, dtype=torch.float, requires_grad=False)
        temp_shifted = torch.tensor(gr[['duration_time','class']].values, dtype=torch.float, requires_grad=False)
        x = pad_sequence([temp], batch_first=True)
        y_truth = pad_sequence([temp_shifted], batch_first=True)
        cur_trace_len = int(x.shape[1])
        if cur_trace_len > max_prefix_len:
            continue
        activites = torch.argmax(x[:, :, events])
        # When we create mini batches, the length of the last one probably is less than the batch size, and it makes problem for the LSTM, therefore we skip it.
        if (x.size()[0] < batch):
            continue
        for cur_prefix_len in range(1,cur_trace_len):
            
            # Separating event and timestamp
            y_truth_timestamp = y_truth[:, cur_prefix_len, 0].view(batch, 1, -1)
            
            y_truth_event = y_truth[:, cur_prefix_len, 1].view(batch, 1, -1)
            
            # Executing LSTM
            if cur_prefix_len == 1:
                x_inn = x[:,  :cur_prefix_len, selected_columns]
                prev_event = x[:,  :cur_prefix_len, len(selected_columns)+1].ravel().detach().numpy().astype("int")
            else:
                x_inn = x[:,  :cur_prefix_len, selected_columns]
                
            rnnG = torch.load("checkpoints/"+dataset_name+"/event_timestamp_prediction/prefix_"+str(cur_prefix_len)+"/rnnG.m")
            rnnG.eval()
            y_pred = rnnG(x_inn[:,  :cur_prefix_len, selected_columns])

            # Just taking the last predicted element from each the batch
            y_pred_last = y_pred[0: batch, cur_prefix_len - 1, :]
            y_pred_last = y_pred_last.view((batch, 1, -1))
            y_pred_last_event = torch.argmax(F.softmax(y_pred_last[:, :, events], dim=2), dim=2)

            #Storing list of predictions and corresponding ground truths (to be used for f1score)
            y_truth_list += list(y_truth_event.flatten().data.cpu().numpy().astype(int))
            y_pred_last_event_list += list(y_pred_last_event.flatten().data.cpu().numpy().astype(int))

            # checking dfg compliance for predicted event
            if not(int(prev_event[0]) == 0 and int(y_pred_last_event.detach()[0][0])==0):
                dfg_compliance_bool = check_dfg_compliance(prev_event , y_pred_last_event.detach(), dset=dataset_name)
                dfg_compliance_pred.append(int(dfg_compliance_bool))

                dfg_compliance_gt_bool = check_dfg_compliance(prev_event , y_truth_event.detach().reshape(y_pred_last_event.shape), dset=dataset_name)
                dfg_compliance_gt.append(int(dfg_compliance_gt_bool))

                if y_pred_last_event.flatten().data.cpu().numpy().astype(int)==y_truth_event.flatten().data.cpu().numpy().astype(int):
                    accuracy.append(1)
                else:
                    accuracy.append(0)

            # Computing MAE for the timestamp
            y_pred_timestamp = y_pred_last[:, :, len(events)].view((batch, 1, -1))
            mae.append(torch.abs(y_truth_timestamp - y_pred_timestamp).mean().detach())
            dur += max(y_pred_timestamp.detach().numpy()[0][0],0)  #adding to total proccess duration, making sure y_pred is non-negative
            dur_gt += max(y_truth_timestamp.detach().numpy()[0][0],0)
            prev_event = y_pred_last_event.ravel().detach().numpy().astype("int")
            
        # GS cases
        try:
            if dur < thresh:
                num_overall_goal_satisfied += 1
            else:
                num_overall_goal_not_satisfied += 1

            if dur_gt < thresh:
                num_overall_goal_satisfied_gt += 1
            else:
                num_overall_goal_not_satisfied_gt += 1
        except Exception as e: print(e)
        dur = 0
        dur_gt = 0
    tot = num_overall_goal_satisfied+ num_overall_goal_not_satisfied
    gs = num_overall_goal_satisfied/tot
    gv = num_overall_goal_not_satisfied/tot
    
    tot_gt = num_overall_goal_satisfied_gt+ num_overall_goal_not_satisfied_gt
    gs_gt = num_overall_goal_satisfied_gt/tot_gt
    gv_gt = num_overall_goal_not_satisfied_gt/tot_gt
    print("compliance pred", np.mean(np.array(dfg_compliance_pred)))
    print("compliance gt", np.mean(np.array(dfg_compliance_gt)))
    print("gs ", gs)
    print("gv ", gv)
    print("gs gt", gs_gt)
    print("gv gt", gv_gt)
    print("acc ",np.mean(np.array(accuracy)))
    print("mae ", np.mean(np.array(mae)))
datasets = ["traffic_ss"]  

for dataset in datasets:
    print("dataset: ",dataset)
    model_eval_test(dataset_name=dataset, csvwriter=None)