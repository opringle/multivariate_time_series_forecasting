#modules
import math
import os
import sys
import numpy as np

#custom modules
import config
import mxnet as mx

##############################################
#load input data
##############################################

if config.q < min(config.filter_list):
    print("\n\n\n\t\tINCREASE q...")

#read in multivariate time series data
x = np.load("../data/electric.npy")

if config.max_training_examples:
    x = x[:config.max_training_examples]

##############################################
# loop through data constructing features/labels
##############################################

#create arrays for storing values in
x_ts = np.zeros((x.shape[0] - config.q,  config.q, x.shape[1]))
y_ts = np.zeros((x.shape[0] - config.q))

#loop through collecting records for ts analysis depending on q
for n in list(range(x.shape[0])):

    if n + 1 < config.q:
        continue

    if n + 1 + config.horizon > x.shape[0]:
        continue

    else:
        y_n = x[n+config.horizon,config.predict_index]
        x_n = x[n+1 - config.q:n+1,:]

    x_ts[n - config.q] = x_n
    y_ts[n - config.q] = y_n

#split into training and testing data
training_examples = int(x_ts.shape[0] * config.split[0])

x_train = x_ts[:training_examples]
y_train = y_ts[:training_examples]
x_test = x_ts[training_examples:]
y_test = y_ts[training_examples:]

print("\nsingle X, Y record example:\n\n")
print(x_train[:1])
print(y_train[:1])
print(x_train[1:2])
print(y_train[1:2])

print("\ntraining examples: ", x_train.shape[0], "\n\ntest examples: ", x_test.shape[0])

###############################################
#define input data iterators for training and testing
###############################################

train_iter = mx.io.NDArrayIter(data={'seq_data': x_train},
                               label={'seq_label': y_train},
                               batch_size=config.batch_size)

val_iter = mx.io.NDArrayIter(data={'seq_data': x_test},
                             label={'seq_label': y_test},
                             batch_size=config.batch_size)

#print input shapes
input_feature_shape = train_iter.provide_data[0][1]
input_label_shape = train_iter.provide_label[0][1]
print("\nfeature input shape: ", input_feature_shape,
      "\nlabel input shape: ", input_label_shape)


####################################
# define neural network graph
####################################

#create placeholders to refer to when creating network graph (names are defined in data iterators)
seq_data = mx.symbol.Variable(train_iter.provide_data[0].name)
seq_label = mx.sym.Variable(train_iter.provide_label[0].name)

# scale input data so features are all between 0 and 1
normalized_seq_data = mx.sym.BatchNorm(data = seq_data)

# reshape data before applying convolutional layer (takes 4D shape incase you ever work with images)
conv_input = mx.sym.Reshape(data=normalized_seq_data, shape=(config.batch_size, 1, config.q, x.shape[1]))


print("\n\t#################################\n\
       #convolutional component:\n\
       #################################\n")

#create many convolutional filters to slide over the input
outputs = []
for i, filter_size in enumerate(config.filter_list):

        # zero pad the input array, adding rows at the bottom only
        # this ensures the number output rows = number input rows after applying kernel
        padi = mx.sym.pad(data=conv_input, mode="constant", constant_value=0, 
                            pad_width=(0, 0, 0, 0, 0, filter_size - 1, 0, 0))                  
        padi_shape = padi.infer_shape(seq_data=input_feature_shape)[1][0]

        # apply convolutional layer (the result of each kernel position is a single number)
        convi = mx.sym.Convolution(data=padi, kernel=(filter_size, x.shape[1]), num_filter=config.num_filter)
        convi_shape = convi.infer_shape(seq_data=input_feature_shape)[1][0]

        #apply relu activation function as per paper
        acti = mx.sym.Activation(data=convi, act_type='relu')

        #transpose output to shape in preparation for recurrent layer (batches, q, num filter, 1)
        transposed_convi = mx.symbol.transpose(data=acti, axes= (0,2,1,3))
        transposed_convi_shape = transposed_convi.infer_shape(seq_data=input_feature_shape)[1][0]

        #reshape to (batches, q, num filter)
        reshaped_transposed_convi = mx.sym.Reshape(data=transposed_convi, target_shape=(config.batch_size, config.q, config.num_filter))
        reshaped_transposed_convi_shape = reshaped_transposed_convi.infer_shape(seq_data=input_feature_shape)[1][0]

        #append resulting symbol to a list
        outputs.append(reshaped_transposed_convi)

        print("\n\tpadded input size: ", padi_shape)
        print("\n\t\tfilter size: ", (filter_size, x.shape[1]), " , number of filters: ", config.num_filter)
        print("\n\tconvi output layer shape (notice length is maintained): ", convi_shape)
        print("\n\tconvi output layer after transposing: ", transposed_convi_shape)
        print("\n\tconvi output layer after reshaping: ", reshaped_transposed_convi_shape)

#concatenate symbols to (batch, total_filters, q, 1)
conv_concat = mx.sym.Concat(*outputs, dim=2)
conv_concat_shape = conv_concat.infer_shape(seq_data=input_feature_shape)[1][0]
print("\nconcat output layer shape: ", conv_concat_shape)

#calculate the total number of convolutional filters, each will produce a single number
total_filters = config.num_filter * len(config.filter_list * convi_shape[2])

print("\n\t#################################\n\
       #recurrent component:\n\
       #################################\n")

#define a gated recurrent unit cell, which we can unroll into many symbols based on our desired time dependancy
cell_outputs = []
for i, recurrent_cell in enumerate(config.rcells):

    #unroll the lstm cell, obtaining a symbol each time
    # Each symbol is of shape (batch_size, hidden_dim)
    outputs, states = recurrent_cell.unroll(length=conv_concat_shape[1], inputs=conv_concat, merge_outputs=False, layout="NTC")

    #for each unrolled timestep
    step_outputs = []
    for i, step_output in enumerate(outputs):

        #apply relu activation function
        acti = mx.sym.Activation(data=step_output, act_type='relu')

        if i == 0:
            print("\n\teach of the ", conv_concat_shape[1], " unrolled hidden cells in the RNN is of shape: ",
                  step_output.infer_shape(seq_data=input_feature_shape)[1][0], "\n")

        #append symbol to a list
        step_outputs.append(step_output)

    #concatenate output for each timestep (shape is now (batch_size, state size * unrolls))
    concatenated_output = mx.sym.concat(*step_outputs, dim=1)

    #append to list
    cell_outputs.append(concatenated_output)

print("\n\tconcatenated recurrent shape for each of the ", len(config.rcells),
      " RNN cell types: ", concatenated_output.infer_shape(seq_data=input_feature_shape)[1][0], "\n")


#concatenate output from each type of recurrent cell (shape is now (batch_size, len(config.cells) * statesize * unrolls)) 
rnn_component = mx.sym.concat(*cell_outputs, dim=1)
print("\nshape after combining RNN cell types: ", rnn_component.infer_shape(seq_data=input_feature_shape)[1][0])

print("\n\t#################################\n\
       #recurrent-skip component:\n\
       #################################\n")

# connect hidden cells that are a defined time interval apart,
# because in practice very long term dependencies are not captured by LSTM/GRU
# eg if you are predicting electricity consumption you want to connect data 24hours apart
# and if your data is every 60s you make a connection between the hidden states at time t and at time t + 24h
# this connection would not be made by an LSTM since 24 is so many hidden states away

#define number of cells to skip through to get a certain time interval back from current hidden state
p =int(config.seasonal_period / config.time_interval)
print("adding skip connections for cells ", p, " intervals apart...")

#define a gated recurrent unit cell, which we can unroll into many symbols based on our desired time dependancy
skipcell_outputs = []
for i, recurrent_cell in enumerate(config.skiprcells):

    #unroll the rnn cell, obtaining a symbol each time
    # Each symbol is of shape (batch_size, hidden_dim)
    outputs, states = recurrent_cell.unroll(length=conv_concat_shape[1], inputs=conv_concat, merge_outputs=False, layout="NTC")

    #for each unrolled timestep
    counter = 0
    step_outputs = []
    for i, current_cell in enumerate(outputs):

        #try adding a concatenated skip connection
        try:

            #get seasonal cell p steps behind
            skip_cell = outputs[i + p]

            #connect this cell to is seasonal neighbour
            cell_pair = [current_cell, skip_cell]
            concatenated_pair = mx.sym.concat(*cell_pair, dim=1)

            #apply relu activation
            acti = mx.sym.Activation(data=concatenated_pair, act_type='relu')

            #append symbol to a list
            step_outputs.append(concatenated_pair)

            counter += 1

        except IndexError:
            #if we cannot concatenate just leave as is

            #apply relu activation
            acti = mx.sym.Activation(data=current_cell, act_type='relu')

            #append symbol to a list
            step_outputs.append(current_cell)

    #concatenate output for each timestep (shape is now (batch_size, state size * unrolls))
    concatenated_output = mx.sym.concat(*step_outputs, dim=1)

    #append to list
    skipcell_outputs.append(concatenated_output)

    print("\n\t", counter, " skip connections created...")
    print("\n\t each of the ", counter, " pairs of connected hidden cells is of shape: ",
            concatenated_pair.infer_shape(seq_data=input_feature_shape)[1][0])
    print("\n\teach of the ", conv_concat_shape[1] - counter, " remaining unrolled hidden cells in the skip-RNN is of shape: ",
                current_cell.infer_shape(seq_data=input_feature_shape)[1][0])
    print("\n\tresulting shape should be", (config.batch_size, (counter) *(config.recurrent_state_size * 2) + 
                                            (conv_concat_shape[1] - counter) * config.recurrent_state_size))


print("\n\tconcatenated unrolled recurrent shape for each of the ", len(config.skiprcells), 
    " combined skip pairs after adding connections: ", concatenated_output.infer_shape(seq_data=input_feature_shape)[1][0], "\n")

#concatenate output from each type of recurrent cell (shape is now (batch_size, len(config.cells) * statesize * unrolls)) 
skiprnn_component = mx.sym.concat(*skipcell_outputs, dim=1)
print("\ncombined flattened recurrent-skip shape : ", skiprnn_component.infer_shape(seq_data=input_feature_shape)[1][0])

print("\n\t#################################\n\
       #autoregressive component:\n\
       #################################\n")

#AR component simply says the next prediction 
# is the some constant times all the previous values available (q).

#pass recurrent layer to fully connected layer with q neurons (same as AR)
ar_component = mx.sym.FullyConnected(data=normalized_seq_data, num_hidden=config.q)
print("\nautoregressive layer output shape: ", ar_component.infer_shape(seq_data=input_feature_shape)[1][0])

#pass to fully connected layer to map to a single value
ar_output = mx.sym.FullyConnected(data=ar_component, num_hidden=1)
print("\nAR output shape : ", ar_output.infer_shape(seq_data=input_feature_shape)[1][0])

#do not apply activation function since we want this to be linear

print("\n\t#################################\n\
       #combine AR and NN components:\n\
       #################################\n")

#combine model components
neural_components = mx.sym.concat(*[rnn_component, skiprnn_component], dim = 1)

#pass to fully connected layer to map to a single value
neural_output = mx.sym.FullyConnected(data=neural_components, num_hidden=1)
print("\nNN output shape : ", neural_output.infer_shape(seq_data=input_feature_shape)[1][0])

#sum the output from AR and deep learning
model_output = neural_output + ar_output
print("\nshape after adding autoregressive output: ", model_output.infer_shape(seq_data=input_feature_shape)[1][0])  

#########################################
# loss function
#########################################

#compute the gradient of the L2 loss
loss_grad = mx.sym.LinearRegressionOutput(data = model_output, label = seq_label)

#set network point to back so name is interpretable
batmans_NN = loss_grad

#########################################
# create a trainable module on CPU/GPU
#########################################

model = mx.mod.Module(symbol=batmans_NN,
                      context=config.context,
                      data_names=[v.name for v in train_iter.provide_data],
                      label_names=[v.name for v in train_iter.provide_label])


####################################
#define evaluation metrics to show when training
#####################################

eval_metric_1 = mx.metric.RMSE()

################
# #fit the model
################

# allocate memory given the input data and label shapes
model.bind(data_shapes=train_iter.provide_data,
           label_shapes=train_iter.provide_label)

# initialize parameters by uniform random numbers
model.init_params()

# use SGD with learning rate 0.1 to train
model.init_optimizer(optimizer='sgd', optimizer_params=(
    ('learning_rate', config.learning_rate), ))

# train n epochs, i.e. going over the data iter one pass
for epoch in range(config.num_epoch):
    train_iter.reset()
    val_iter.reset()
    eval_metric_1.reset()
    for batch in train_iter:
        model.forward(batch, is_train=True)       # compute predictions
        # accumulate prediction accuracy
        model.update_metric(eval_metric_1, batch.label)
        model.backward()                          # compute gradients
        model.update()                            # update parameters

    print('Epoch %d, Training %s' % (epoch, eval_metric_1.get()))

    eval_metric_1.reset()
    for batch in val_iter:
        model.forward(batch, is_train=False)       # compute predictions
        # accumulate prediction accuracy
        model.update_metric(eval_metric_1, batch.label)

    print('Epoch %d, Validation %s' % (epoch, eval_metric_1.get()))


################
# save model after epochs
################

model.save_checkpoint(
    prefix='my_model',
    epoch=config.num_epoch,
    save_optimizer_states=False,
)
