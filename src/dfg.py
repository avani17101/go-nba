import numpy as np

def get_dfg_graph(dset):
    """
    return the dfg graph for a dataset
    """
    # helpesk dfg graph
    graphs = {}
    graph = {}
    graph[0] = {1:3644, 3:108, 0:1}  #end of process instance 1, start of instance 2
    graph[3] = {1:105}
    graph[1] = {8:3483,1:394}
    graph[8] = {2:42, 9:851, 6:4150,4:9, 7:4} #edges:weights
    graph[2] = {5:3, 6:37}
    graph[9] = {6:382,8:4270}
    graph[4] = {6:8}
    graph[7] = {6:3}
    graph[5] = {6:4}
    graph[6] = {0:3804} #end
    graphs['helpdesk'] = graph


    
    graph2 = {0: {1:4739, 2:67,3:4852, 0:1}, 1: {3:2515},2:{0:57},3:{3:18942,5:5015,0:2355},4:{4:9676,6:1731},5:{5:19767,6:3209},6:{6:5144,4:21,0:2751}}
    graphs['bpi_12_w'] = graph2

    graph3 = {0:{1:15037},1:{2:10356, 6:4678, 0:1},2:{3:8036,0:2037},3:{4:7278, 7:325, 11:26},4:{5:5772,8:277},5:{0:5490},6:{0:6730},7:{4:278},8:{0:306,9:61},9:{10:92},10:{6:40},11:{4:24}}
    graphs['traffic_ss'] = graph3

    # sampled graph
    graph4 = {
        0: {1:988, 0:1},
        1: {5:963},
        2: {3:1608,2:386},
        3: {10:320, 13:19,14:13,15:12,16:3,2:1310},
        4: {7:122},
        5: {6:896},
        6: {4:148, 7:280},
        7: {8:497},
        8: {9:485, 12:44},
        9: {2:398, 3:364},
        10: {11:272,0:392},
        11: {0:287},
        12: {4: 37},
        13: {0: 53},
        14: {0: 18},
        15: {0: 13},
        16: {0: 6}
        }
    # graphs['sepsis_ss'] = graph4
    full_graph = {
        0: {1:995, 0:1},
        1: {5:963},
        2: {3:1608,2:386},
        3: {10:320, 13:19,14:13,15:12,16:3,2:1310,4:629},
        4: {3:122},
        5: {6:896},
        6: {7:280},
        7: {8:497},
        8: {9:485, 12:44},
        9: {2:398, 3:364},
        10: {11:272,0:392},
        11: {0:287},
        12: {4: 37},
        13: {0: 53},
        14: {0: 18},
        15: {0: 13},
        16: {0: 6}
        }
    graphs['sepsis_ss'] = full_graph  
    # graph6 = {
    #     0: {1:1},
    #     1: {5:1, 7:1,6:1,3:1,12:1,2:1,4:1, 9:1, 13:1, 10:1, 14:1, 15:1,16:1,11:1},
    #     2: {9:1,13:1},
    # }
    graphs["bpi2019"] = {
        0: {30:1,1:1,28:1, 0:1},
        1:{2:1,15:1, 16:1,21:1,17:1, 29:1, 23:1, 8:1,24:1,25:1,11:1},
        2:{3:1,7:1},
        3:{0:1,27:1,4:1,10:1,19:1,20:1},
        4:{0:1,6:1},
        5:{0:1},
        6:{0:1,5:1},
        7:{0:1},
        8:{4:1},
        9:{0:1},
        10:{4:1},
        11:{12:1},
        12:{14:1},
        13:{14:1,18:1},
        14:{13:1},
        15:{2:1},
        16:{0:2,31:1,22:1},
        17:{2:1,9:1},
        18:{11:1},
        19:{4:1},
        20:{4:1},
        21:{26:1,2:1},
        22:{3:1},
        23:{16:1},
        24:{0:1},
        25:{24:1},
        26:{2:1},
        27:{0:1},
        28:{1:1,32:1},
        29:{2:1},
        30:{15:1,2:1,28:1},
        31:{22:1},
        32:{1:1}
    }
    return graphs[dset]

def checkCandidate(instance, dset="helpdesk"):
    '''
    check if a process activity sequence is compliant to the dfg model 
    args: 
        instance: process activity sequence
        returns: True, if compliant
                 False, if non-compliant
    '''
    # print("dest",dset)
    graph = get_dfg_graph(dset)
    nodes_comply = 0
    # print("instance",instance)
    # if (instance[0] not in graph[0].keys()): #first activity must start from node 1
    #     return False
    
    for i in range(1,len(instance)):
        prev_node = instance[i-1]
        available_nodes = list(graph[prev_node].keys())
        # print(available_nodes)
        cur_node = instance[i]
        # if cur_node is in available_nodes(prev_node): comply
        if(cur_node in available_nodes):
            nodes_comply += 1  
    if(nodes_comply == len(instance)-1):
        return True
    else:
        return False
    
def check_dfg_compliance(act, y, dset):
    '''
    check if a process activity [act, y] sequence is compliant to the dfg model 
    args: 
        act: current gt process activity sequence
        y: predicted activity
        returns: True, if compliant
                 False, if non-compliant
    ''' 
    seq = np.array([act, y])
    # seq = np.array(seq, dtype=int)
    valid = checkCandidate(seq, dset)
    return valid

# def check_dfg_compliance(act, y, dset):
#     '''
#     check if a process activity [act, y] sequence is compliant to the dfg model 
#     args: 
#         act: current gt process activity sequence
#         y: predicted activity
#         returns: True, if compliant
#                  False, if non-compliant
#     ''' 
#     seq = np.append(act, y.numpy()[0])
#     seq = np.array(seq, dtype=int)
#     valid = checkCandidate(seq, dset)
#     return valid
