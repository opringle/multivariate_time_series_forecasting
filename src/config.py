import mxnet as mx

q = 3 #number of timesteps used to make a prediction
split = [0.8,0.2] #the ration of training to testing data
batch_size = 10 #number of examples to pass into the network/use to compute gradient of the loss function
filter_list = [2,2] #must be smaller than q!!!, size of filters sliding over the input data
num_filter = 10 #number of each filter size
recurrent_state_size = 92 #number of hidden units for each unrolled recurrent layer
context = mx.cpu() #train on cpu because maclyfe
learning_rate = 0.1 #learning rate for plain vanilla sgd
num_epoch = 100 #how many times to backpropogate and update weights
time_interval = 24*60*60 #seconds between feature values (data defined)
seasonal_period = 24*60*60 #seconds between important measurements (choose)

#choose recurrent cells for the recurrent layer
rcells = [mx.rnn.GRUCell(num_hidden=recurrent_state_size)]

#choose recurrent cells for the recurrent_skip layer
skiprcells = [mx.rnn.GRUCell(num_hidden=recurrent_state_size)]
