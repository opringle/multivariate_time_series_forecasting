import mxnet as mx
import math

#input data parameters
max_training_examples = None #limit the number of training examples used
split = [0.6, 0.2, 0.2] #the proportion of training, validation and testing examples
horizon = 3 #how many time steps ahead do we wish to predict?
time_interval = 60*60 #seconds between feature values (data defined)

#model hyperparameters
batch_size = 128 #number of examples to pass into the network/use to compute gradient of the loss function
num_epoch = 100 #how many times to backpropogate and update weights
seasonal_period = 24*60*60 #seconds between important measurements (tune)
q = max(24*7, math.ceil(seasonal_period / time_interval))  # windowsize used to make a prediction
filter_list = [6] #must be smaller than q!!!, size of filters sliding over the input data
num_filter = 50 #number of each filter size
recurrent_state_size = 50 #number of hidden units for each unrolled recurrent layer
recurrent_skip_state_size = 20  # number of hidden units for each unrolled recurrent layer
optimizer = 'Adam'
optimizer_params = {'learning_rate': 0.001,
                    'beta1': 0.9,
                    'beta2': 0.999}
dropout = 0.1 #dropout probability after convolutional/recurrent and autoregressive layers
rcells = [mx.rnn.GRUCell(num_hidden=recurrent_state_size)] #choose recurrent cells for the recurrent layer
skiprcells = [mx.rnn.GRUCell(num_hidden=recurrent_skip_state_size, prefix="skip_")] #choose recurrent cells for the recurrent_skip layer

#computational parameters
context = mx.cpu() #train on cpu because maclyfe
