from numpy.random import seed
seed(1)
import tensorflow
# tensorflow.random.set_random_seed(1)
tensorflow.random.set_seed(1)
from argparse import ArgumentParser
from datetime import datetime
import numpy as np
from os import environ

from BasicTimeEnvironment import Process_Environment
from BasicTimeProcessor import MyProc

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM
from tensorflow.keras.optimizers import Adam

from rl.agents import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import GreedyQPolicy,BoltzmannQPolicy
import os

environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
environ["CUDA_VISIBLE_DEVICES"]="1"

ENV_NAME = 'time_considering'
dataset_name = "Helpdesk"
parser = ArgumentParser()
parser.add_argument("--memlen", type = int, default=1, help="MEMORY_LENGTH")
opt = parser.parse_args()
MEMORY_LENGTH = opt.memlen

# for MEMORY_LENGTH in [1]:#,6,8#2,3,4,5,6,7,8,10,20
data_directory = "../../dataset/"+dataset_name+"/"
result_directory = "../../dataset/"+dataset_name+"/result/"+str(MEMORY_LENGTH)+"/time/"


if not os.path.isdir(result_directory):
    os.makedirs(result_directory)

start_time = datetime.now()
# [0, 1, 10, 60, 120, 240, 480, 1440, 2880, 4320, 7200, 10080, 15120, 20160, 30240, 40320, 50400] BPI12 (sul paper ci sono mi pare se hai dubbi)
# [0, 1, 10, 60, 120, 240, 480, 1440, 2880, 4320, 7200, 10080, 15120, 20160, 30240, 40320, 50400, 60480,80640] Helpdesk
# [0, 1440, 10080, 20160, 30240, 40320, 80640, 120960, 241920, 483840, 725760, 967680, 1451520, 1935360, 2419200, 5806080] RoadFines

bins_ranges =[0, 1, 10, 60, 120, 240, 480, 1440, 2880, 4320, 7200, 10080, 15120, 20160, 30240, 40320, 50400, 60480,80640]

processor = MyProc(data_directory=data_directory,bins_ranges= bins_ranges)

nb_events,nb_bins = processor.get_nb_acts()
bins_ranges = processor.get_bins_ranges()

env = Process_Environment(testing = False,training_data = True,data_directory=data_directory,result_directory=result_directory, bins_ranges = bins_ranges , window_size = MEMORY_LENGTH, processed_acts_num = nb_events)
#test_env=Process_Environment(testing = True, training_data = True,data_directory=data_directory,result_directory=result_directory, bins_ranges = bins_ranges, window_size = MEMORY_LENGTH) #, processed_acts_num = nb_events)

test_data = Process_Environment(testing = True, training_data = False,data_directory=data_directory,result_directory=result_directory, bins_ranges = bins_ranges, window_size = MEMORY_LENGTH, processed_acts_num = nb_events)

print("bins available : "+str(nb_bins))

processed_observation_space = nb_events + 3 #numero di features temporali prestabilite(stanno in input proprio)

MyModel = Sequential()
MyModel.add(LSTM(units=150, activation='relu', input_shape=(MEMORY_LENGTH, processed_observation_space), return_sequences=True))
MyModel.add(LSTM(units=150, activation='relu', return_sequences=False))
MyModel.add(Dense(units=nb_bins,activation='linear'))

memory = SequentialMemory(limit=1000000, window_length=MEMORY_LENGTH)
#memory = PrioritizedMemory(limit=300000, alpha=.6, start_beta=.2, end_beta=9., steps_annealed=20000, window_length=MEMORY_LENGTH)


dqn = DQNAgent(model=MyModel, nb_actions=nb_bins, policy=BoltzmannQPolicy(clip=(-15.,15)), test_policy=GreedyQPolicy() , memory=memory,
                processor=processor, enable_double_dqn=True, enable_dueling_network=True, nb_steps_warmup=50, gamma=.9,
                target_model_update=0.01,train_interval=1, delta_clip=1.)

lr = .001
# if type(memory) == PrioritizedMemory:
#     lr/= 4#4
dqn.compile(Adam(lr=lr),metrics=['mae'])

# dqn.load_weights(result_directory+'DQN_time.h5f')
begin_train_time = datetime.now()
for trance in [1,2,3,4,5,6,7]:
    dqn.fit(env, nb_steps=100000,nb_max_start_steps=MEMORY_LENGTH-1, visualize=True, verbose=1)
    dqn.save_weights(result_directory+'DQN_time'+str(trance)+'.h5f', overwrite=True)

end_train_time = datetime.now()

episodes = test_data.get_episodes_num()

dqn.test(test_data, nb_episodes=episodes,nb_max_start_steps=MEMORY_LENGTH-1, visualize=True)

end_test_time = datetime.now()

f = open(result_directory + "timing.txt", "a")
f.write("\nrecord date\n")
f.write(str(datetime.now()))
f.write("\ntime to setup\n")
f.write(str(begin_train_time - start_time))
f.write("\ntime to train\n")
f.write(str(end_train_time - begin_train_time))
f.write("\ntime to test\n")
f.write(str(end_test_time - end_train_time))
f.close()

