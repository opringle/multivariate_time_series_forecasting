#modules
import mxnet as mx
import numpy as np
import sys
import os
import math

#custom modules
import config


##############################################
#load input data
##############################################

if config.q < min(config.filter_list):
    print("\n\n\n\t\tINCREASE q...")

#read in features and labels
x = np.load("../data/x.npy")
y = np.load("../data/y.npy")

##############################################
# loop through data collecting features for previous q days
##############################################

#loop through collecting records for ts analysis depending on q
for n in list(range(x.shape[0])):
    if n + 1 < config.q:
        continue
    else:
        y_n = y[n,:]
        x_n = x[n+1 - config.q:n+1,:]

    if n + 1 == config.q:
        x_ts = np.zeros((y.shape[0] - config.q,  config.q, x.shape[1]))
        y_ts = np.zeros((y.shape[0] - config.q,  x.shape[1]))

    x_ts[n - config.q] = x_n
    y_ts[n - config.q] = y_n

#split into training and testing data
training_examples = int(x_ts.shape[0] * config.split[0])

x_train = x_ts[:training_examples]
y_train = y_ts[:training_examples]
x_test = x_ts[training_examples:]
y_test = y_ts[training_examples:]

# print("\n\n\n\n")
# print(x_train[:4])
# print(y_train[:4])

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


# reshape embedded data for next layer takes 4D shape incase you ever work with images
conv_input = mx.sym.Reshape(data=seq_data, shape=(config.batch_size, 1, config.q, y.shape[1]))


#######################################
# convolutional component
#######################################

#create many convolutional filters to slide over the input
outputs = []
for i, filter_size in enumerate(config.filter_list):

        # zero pad the input array, adding rows at the bottom only, to ensure old shape is maintained
        padi = mx.sym.pad(data=conv_input, mode="constant", constant_value=0, 
                            pad_width=(0, 0, 0, 0, 0, config.q - filter_size, 0, 0))
                            
        padi_shape = padi.infer_shape(seq_data=input_feature_shape)[1][0]

        # apply convolutional layer (the result of each kernel position is a single number)
        convi = mx.sym.Convolution(data=padi, kernel=(filter_size, y.shape[1]), num_filter=config.num_filter)
        convi_shape = convi.infer_shape(seq_data=input_feature_shape)[1][0]
        # output data shape: batch_size, channel = num_filter, out_height = see formula, out_width = see formula).

        #apply relu activation function
        acti = mx.sym.Activation(data=convi, act_type='relu')

        #append resulting symbol to a list
        outputs.append(acti)

print("\npadded size: ", padi_shape)
print("\n\tfilter size: ", (filter_size, y.shape[1]), " , number filters per size: ", config.num_filter, "sizes: ", len(config.filter_list))
print("\nconvi output layer shape: ", convi_shape)

#concatenate symbols to (batch, total_filters, hmm,hmm)
conv_concat = mx.sym.Concat(*outputs, dim=1)
print("\nconcat output layer shape: ", conv_concat.infer_shape(seq_data=input_feature_shape)[1][0])

#calculate the total number of convolutional filters, each will produce a single number
total_filters = config.num_filter * len(config.filter_list * convi_shape[2])

#flatten the square results from the conv layer into a single 1D array per batch
conv_flat = mx.sym.Reshape(data=conv_concat, target_shape=(config.batch_size, total_filters))
print("\nflattened shape: ", conv_flat.infer_shape(seq_data=input_feature_shape)[1][0])

#####################################
# recurrent component
#####################################

#define a gated recurrent unit cell, which we can unroll into many symbols based on our desired time dependancy
cell_outputs = []
for i, recurrent_cell in enumerate(config.rcells):

    #unroll the lstm cell, obtaining a symbol each time
    # Each symbol is of shape (batch_size, hidden_dim)
    outputs, states = recurrent_cell.unroll(length=total_filters, inputs=conv_flat, merge_outputs=False, layout="NTC")

    #for each unrolled timestep
    step_outputs = []
    for i, step_output in enumerate(outputs):
        #apply dropout to the lstm output
        #drop = mx.sym.Dropout(data=step_output, p=config.dropout, mode='training')
        #add a fully connected layer with num_neurons = num_possible_tags
        #print("\nrecurrent shape at time: ", i, step_output.infer_shape(seq_data=input_feature_shape)[1][0])
        #fc = mx.sym.FullyConnected(data=step_output, num_hidden=y.shape[1])
        #print("\nfully connected shape at time: ", i, fc.infer_shape(seq_data=input_feature_shape)[1][0])

        #apply relu activation function
        acti = mx.sym.Activation(data=step_output, act_type='relu')

        #append symbol to a list
        step_outputs.append(acti)

    #concatenate output for each timestep (shape is now (batch_size, state size * unrolls))
    concatenated_output = mx.sym.concat(*step_outputs, dim=1)

    #append to list
    cell_outputs.append(concatenated_output)

print("\nconcatenated unrolled recurrent shap for each of the ", len(config.rcells), " cell types: ", concatenated_output.infer_shape(seq_data=input_feature_shape)[1][0])

#concatenate output from each type of recurrent cell (shape is now (batch_size, len(config.cells) * statesize * unrolls)) 
concatenated_rnn_outputs = mx.sym.concat(*cell_outputs, dim=1)
print("\nflattened recurrent shape : ", concatenated_rnn_outputs.infer_shape(seq_data=input_feature_shape)[1][0])

#####################################
# recurrent-skip component
#####################################

# connect hidden cells that are a defined time interval apart,
# because in practice very long term dependencies are not captured by LSTM/GRU
# eg if you are predicting electricity consumption you want to connect data 24hours apart
# and if your data is every 60s you make a connection between the hidden states at time t and at time t + 24h
# this connection would not be made by an LSTM since 24 is so many hidden states away

#define number of cells to skip through to get a certain time interval back from current hidden state
p =int(config.seasonal_period / config.time_interval)

#define a gated recurrent unit cell, which we can unroll into many symbols based on our desired time dependancy
skipcell_outputs = []
for i, recurrent_cell in enumerate(config.skiprcells):

    #unroll the rnn cell, obtaining a symbol each time
    # Each symbol is of shape (batch_size, hidden_dim)
    outputs, states = recurrent_cell.unroll(length=total_filters, inputs=conv_flat, merge_outputs=False, layout="NTC")

    #for each unrolled timestep
    step_outputs = []
    for i, current_cell in enumerate(outputs):

        if i + 1 < len(outputs):

            #get seasonal cell p steps behind
            skip_cell = outputs[i + p]

            #connect this cell to is seasonal neighbour
            cell_pair = [current_cell, skip_cell]
            concatenated_output = mx.sym.concat(*cell_pair, dim=1)

            #apply relu activation
            acti = mx.sym.Activation(data=concatenated_output, act_type='relu')

            #append symbol to a list
            step_outputs.append(acti)

    #concatenate output for each timestep (shape is now (batch_size, state size * unrolls))
    concatenated_output = mx.sym.concat(*step_outputs, dim=1)

    #append to list
    skipcell_outputs.append(concatenated_output)

print("\nconcatenated unrolled recurrent shape for each of the ", len(config.rcells), " combined skip pairs after adding connections: ", concatenated_output.infer_shape(seq_data=input_feature_shape)[1][0])

#concatenate output from each type of recurrent cell (shape is now (batch_size, len(config.cells) * statesize * unrolls)) 
concatenated_skiprnn_outputs = mx.sym.concat(*cell_outputs, dim=1)
print("\nflattened recurrent-skip shape : ", concatenated_skiprnn_outputs.infer_shape(seq_data=input_feature_shape)[1][0])

#########################################
# fully connected/reshaping
#########################################

#concatenate skip and normal rnn outputs
rnn_weights = mx.sym.concat(*[concatenated_rnn_outputs, concatenated_skiprnn_outputs], dim=1)

#pass recurrent layer to fully connected layer
fc = mx.sym.FullyConnected(data=rnn_weights, num_hidden=y.shape[1] * 2)
print("\nfully connected shape : ", fc.infer_shape(seq_data=input_feature_shape)[1][0])

#reshape before applying loss layer so we can predict each class
reshaped_fc = mx.sym.Reshape(data=fc, target_shape=(config.batch_size, 2, y.shape[1]))
print("\nreshaped connected shape: ", reshaped_fc.infer_shape(seq_data=input_feature_shape)[1][0])                                                              

#########################################
# loss function
#########################################

#compute the gradient of the softmax loss , applying softmax to axis 1 of each batch
loss_grad = mx.sym.SoftmaxOutput(data=reshaped_fc, label=seq_label, multi_output=True)

#set network point to back
batmans_NN = loss_grad

# create a trainable module on CPU/GPU
model = mx.mod.Module(symbol=batmans_NN,
                      context=config.context,
                      data_names=[v.name for v in train_iter.provide_data],
                      label_names=[v.name for v in train_iter.provide_label])


####################################
#define evaluation metrics to show when training
#####################################

eval_metric_1 = mx.metric.Accuracy()

################
# #fit the model
################

# model.fit(
#     train_data=train_iter,
#     eval_data=val_iter,
#     eval_metric=eval_metrics,
#     optimizer='sgd',
#     optimizer_params={"learning_rate": config.learning_rate},
#     num_epoch=config.num_epoch)

###############
# fit the model using lower level code (to debug issues)
##############

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
