from argparse import ArgumentParser
import matplotlib.pyplot as plt
import numpy as np
import event_timestamp_prediction as etp
import event_prediction as ep
import timestamp_prediction as tp
import preparation as pr
import torch
import multiprocessing
import multiprocessing.pool
import csv

device = torch.device('cpu')

if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")


def run(path, dcr_path, mode, design_mat_path, prefix=4, epoch=1, batch_size=1):
    '''
    run the train/val/test code
    args: 
        path: The path to the CSV file
        prefix: Size of the prefix
        epoch:  Number of epochs
    returns:
    '''
    if(mode == "timestamp_prediction"):
        obj = pr.Input()
        #Preparing the input
        obj.run(path = path ,dcr_path=dcr_path, design_mat_path = design_mat_path, prefix= prefix, batch_size= batch_size, mode=mode)

        #Initializing a generator
        selected_columns = obj.selected_columns
        print("Selected columns:", selected_columns)
        obj.selected_columns = selected_columns
        rnnG = tp.LSTMGenerator(seq_len = obj.prefix_len+1, input_size = len(selected_columns), batch = obj.batch, hidden_size= 1 , num_layers = 2, num_directions = 1)
        optimizerG = torch.optim.Adam(rnnG.parameters(), lr=0.0002, betas=(0.5, 0.999))

        #Initializing a discriminator
        rnnD = tp.LSTMDiscriminator(seq_len = obj.prefix_len+2, input_size = len(selected_columns), batch = obj.batch, hidden_size = 1, num_layers =2, num_directions = 1)
        optimizerD = torch.optim.Adam(rnnD.parameters(), lr=0.0002, betas=(0.5, 0.999))

        #Training and testing
        ## comment below if only want to test
        tp.train(rnnD=rnnD, rnnG=rnnG,optimizerD=optimizerD, optimizerG = optimizerG, obj=obj, epoch=epoch)
        #Loading the model from the validation
        rnng_validation = torch.load(obj.path+"/rnnG(validation).m")
        for i in range(1):
            tp.model_eval_test(modelG= rnnG, mode='test', obj = obj)
            tp.model_eval_test(modelG= rnng_validation, mode='test', obj = obj)
            obj.train_valid_test_index()
            obj.mini_batch_creation(batch=obj.batch)
    
    

    if(mode == 'event_timestamp_prediction'):
        
        obj = pr.Input()
        # Preparing the input
        obj.run(path = path ,dcr_path=dcr_path, design_mat_path = design_mat_path, prefix= prefix, batch_size= batch_size, mode=mode)
       
        # #Initializing a generator
        rnnG = etp.LSTMGenerator(seq_len = obj.prefix_len, input_size = len(obj.selected_columns), batch = obj.batch, hidden_size= 2*len(obj.selected_columns) , num_layers = 2, num_directions = 1)
        optimizerG = torch.optim.Adam(rnnG.parameters(), lr=0.0002, betas=(0.5, 0.999))

        # # Initializing a discriminator
        rnnD = etp.LSTMDiscriminator(seq_len=obj.prefix_len + 1, input_size=len(obj.selected_columns), batch=obj.batch, hidden_size=2 * len(obj.selected_columns), num_layers=2, num_directions=1)
        optimizerD = torch.optim.Adam(rnnD.parameters(), lr=0.0002, betas=(0.5, 0.999))

        # # Training and testing
        ## comment below if only want to test
        etp.train(rnnD=rnnD, rnnG=rnnG, optimizerD=optimizerD, optimizerG=optimizerG, obj=obj, epoch=epoch)
        
        # Loading the model from the validation
        rnng_validation = torch.load(obj.path + "/rnnG(validation).m")
        for i in range(3):
            etp.model_eval_test(modelG=rnnG, mode='test', obj=obj)
            etp.model_eval_test(modelG=rnng_validation, mode='test', obj=obj)
            obj.train_valid_test_index()
            obj.mini_batch_creation(batch=obj.batch)


    return obj

# def parallise(idx):
#     run(opt.path, opt.dcr_path,mode=opt.mode,design_mat_path=opt.design_mat_path, prefix=idx, epoch=opt.epochs)

parser = ArgumentParser()
parser.add_argument("--path", default='dataset/sepsis_ss.csv', help="path to dataset")
parser.add_argument("--mode", default='timestamp_prediction', help="event_timestamp_prediction or event_prediction or timestamp_prediction")
# parser.add_argument("--check_dcr", default=True, help="whether to check dcr")
parser.add_argument("--design_mat_path", default="dataset/preprocessed/sepsis_ss_design_mat.pkl", help="path to precomputed design matrix")
parser.add_argument("--dcr_path", default='dataset/helpdesk_dcr.xml', help="path to dataset dcr")
parser.add_argument("--prefix_s", type = int, default=1, help="smallest prefix length")
parser.add_argument("--prefix_e", type = int, default=1, help="largest prefix length")
parser.add_argument("--epochs", type = int, default=1, help="num epochs")
parser.add_argument("--batch_size", type = int, default=1, help="batch size")
opt = parser.parse_args()



for i in range(opt.prefix_s, opt.prefix_e+1):
  print(i)
  run(opt.path, opt.dcr_path,mode=opt.mode,design_mat_path=opt.design_mat_path, prefix=i, epoch=opt.epochs,batch_size=opt.batch_size)

