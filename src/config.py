import mxnet as mx
import math

max_training_examples = 1000
horizon = 3 #how many time steps ahead do we wish to predict?
split = [0.6, 0.2, 0.2] #the proportion of training, validation and testing examples
batch_size = 9 #number of examples to pass into the network/use to compute gradient of the loss function
filter_list = [2] #must be smaller than q!!!, size of filters sliding over the input data
num_filter = 1 #number of each filter size
recurrent_state_size = 3 #number of hidden units for each unrolled recurrent layer
context = mx.cpu() #train on cpu because maclyfe
optimizer = 'Adam'
optimizer_params = {'learning_rate': 0.001,
                    'beta1': 0.9,
                    'beta2': 0.999}
num_epoch = 100 #how many times to backpropogate and update weights
time_interval = 15*60 #seconds between feature values (data defined)
seasonal_period = 24*60*60 #seconds between important measurements (choose)
q = max(100, math.ceil(seasonal_period / time_interval))  # number of timesteps used to make a prediction
dropout = 0.1

#choose recurrent cells for the recurrent layer
rcells = [mx.rnn.GRUCell(num_hidden=recurrent_state_size)]

#choose recurrent cells for the recurrent_skip layer
skiprcells = [mx.rnn.GRUCell(num_hidden=recurrent_state_size, prefix="skip_")]
