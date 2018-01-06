#modules
import math
import os
import sys
import numpy as np
import math
import mxnet as mx
import pandas as pd

##############################################
#load input data
##############################################

#read tar.gz data into a pandas dataframe
df = pd.read_csv("./electricity.txt", sep=",", header=None)

#convert to numpy array
x = df.astype(float).as_matrix()

print("\n\tlength of time series: ", x.shape[0])

################################################
# define model hyperparameters and preprocessing parameters
################################################

#input data parameters
max_training_examples = None  # limit the number of training examples used
split = [0.6, 0.2, 0.2] # the proportion of training, validation and testing examples
horizon = 3  # how many time steps ahead do we wish to predict?
time_interval = 60 * 60  # seconds between feature values (data defined)

#model hyperparameters
batch_size = 128  # number of examples to pass into the network/use to compute gradient of the loss function
num_epoch = 100  # how many times to backpropogate and update weights
seasonal_period = 24 * 60 * 60  # seconds between important measurements (tune)
q = max(24 * 7, math.ceil(seasonal_period / time_interval)) # windowsize used to make a prediction
filter_list = [6] #size of each filter we wish to apply
num_filter = 50  # number of each filter size
recurrent_state_size = 50  # number of hidden units for each unrolled recurrent layer
recurrent_skip_state_size = 20 # number of hidden units for each unrolled recurrent-skip layer
optimizer = 'Adam'
optimizer_params = {'learning_rate': 0.001,
                    'beta1': 0.9,
                    'beta2': 0.999}
dropout = 0.1  # dropout probability applied after convolutional/recurrent and autoregressive layers
rcells = [mx.rnn.GRUCell(num_hidden=recurrent_state_size)] #recurrent cell types we wish to use
skiprcells = [mx.rnn.GRUCell(num_hidden=recurrent_skip_state_size, prefix="skip_")] # recurrent-skip cell types we wish to use

#computational parameters
context = mx.cpu()  # train on cpu or gpu


##############################################
# loop through data constructing features/labels
##############################################

#create arrays for storing values in
x_ts = np.zeros((x.shape[0] - q,  q, x.shape[1]))
y_ts = np.zeros((x.shape[0] - q, x.shape[1]))

#loop through collecting records for ts analysis depending on q
for n in list(range(x.shape[0])):

    if n + 1 < q:
        continue

    if n + 1 + horizon > x.shape[0]:
        continue

    else:
        y_n = x[n+horizon,:]
        x_n = x[n+1 - q:n+1,:]

    x_ts[n - q] = x_n
    y_ts[n - q] = y_n

#split into training and testing data
training_examples = int(x_ts.shape[0] * split[0])
valid_examples = int(x_ts.shape[0] * split[1])

x_train = x_ts[:training_examples]
y_train = y_ts[:training_examples]
x_valid = x_ts[training_examples:training_examples + valid_examples]
y_valid = y_ts[training_examples:training_examples + valid_examples]
x_test = x_ts[training_examples + valid_examples:]
y_test = y_ts[training_examples + valid_examples:]

print("\ntraining examples: ", x_train.shape[0],
        "\n\nvalidation examples: ", x_valid.shape[0],
        "\n\ntest examples: ", x_test.shape[0], 
        "\n\nwindow size: ", q,
        "\n\nskip length p: ", seasonal_period / time_interval)


###############################################
#define input data iterators for training and testing
###############################################

train_iter = mx.io.NDArrayIter(data={'seq_data': x_train},
                               label={'seq_label': y_train},
                               batch_size=batch_size)

val_iter = mx.io.NDArrayIter(data={'seq_data': x_valid},
                             label={'seq_label': y_valid},
                             batch_size=batch_size)

test_iter = mx.io.NDArrayIter(data={'seq_data': x_test},
                              label={'seq_label': y_test},
                              batch_size=batch_size)

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

# reshape data before applying convolutional layer (takes 4D shape)
conv_input = mx.sym.Reshape(
    data=seq_data, shape=(batch_size, 1, q, x.shape[1]))


print("\n\t#################################\n\
       #convolutional component:\n\
       #################################\n")

#create many convolutional filters to slide over the input
outputs = []
for i, filter_size in enumerate(filter_list):

        # zero pad the input array, adding rows at the top only
        # this ensures the number output rows = number input rows after applying kernel
        padi = mx.sym.pad(data=conv_input, mode="constant", constant_value=0,
                          pad_width=(0, 0, 0, 0, filter_size - 1, 0, 0, 0))
        padi_shape = padi.infer_shape(seq_data=input_feature_shape)[1][0]

        # apply convolutional layer (the result of each kernel position is a single number)
        convi = mx.sym.Convolution(data=padi, kernel=(
            filter_size, x.shape[1]), num_filter=num_filter)
        convi_shape = convi.infer_shape(seq_data=input_feature_shape)[1][0]

        #apply relu activation function as per paper
        acti = mx.sym.Activation(data=convi, act_type='relu')

        #transpose output shape in preparation for recurrent layer (batches, q, num filter, 1)
        transposed_convi = mx.symbol.transpose(data=acti, axes=(0, 2, 1, 3))
        transposed_convi_shape = transposed_convi.infer_shape(
            seq_data=input_feature_shape)[1][0]

        #reshape to (batches, q, num filter) for recurrent layers
        reshaped_transposed_convi = mx.sym.Reshape(
            data=transposed_convi, target_shape=(batch_size, q, num_filter))
        reshaped_transposed_convi_shape = reshaped_transposed_convi.infer_shape(
            seq_data=input_feature_shape)[1][0]

        #append resulting symbol to a list
        outputs.append(reshaped_transposed_convi)

        print("\n\tpadded input size: ", padi_shape)
        print("\n\t\tfilter size: ", (filter_size,
                                      x.shape[1]), " , number of filters: ", num_filter)
        print("\n\tconvi output layer shape (notice length is maintained): ", convi_shape)
        print("\n\tconvi output layer after transposing: ", transposed_convi_shape)
        print("\n\tconvi output layer after reshaping: ",
              reshaped_transposed_convi_shape)

#concatenate symbols to (batch, total_filters, q, 1)
conv_concat = mx.sym.Concat(*outputs, dim=2)
conv_concat_shape = conv_concat.infer_shape(seq_data=input_feature_shape)[1][0]
print("\nconcat output layer shape: ", conv_concat_shape)

#apply a dropout layer
conv_dropout = mx.sym.Dropout(conv_concat, p=dropout)


print("\n\t#################################\n\
       #recurrent component:\n\
       #################################\n")

#define a gated recurrent unit cell, which we can unroll into many symbols based on our desired time dependancy
cell_outputs = []
for i, recurrent_cell in enumerate(rcells):

    #unroll the lstm cell, obtaining a symbol each time step
    outputs, states = recurrent_cell.unroll(
        length=conv_concat_shape[1], inputs=conv_dropout, merge_outputs=False, layout="NTC")

    #we only take the output from the recurrent layer at time t
    output = outputs[-1]

    #just ensures we can have multiple RNN layer types
    cell_outputs.append(output)

    print("\n\teach of the ", conv_concat_shape[1], " unrolled hidden cells in the RNN is of shape: ",
          output.infer_shape(seq_data=input_feature_shape)[1][0],
          "\nNOTE: only the output from the unrolled cell at time t is used...")


#concatenate output from each type of recurrent cell
rnn_component = mx.sym.concat(*cell_outputs, dim=1)
print("\nshape after combining RNN cell types: ",
      rnn_component.infer_shape(seq_data=input_feature_shape)[1][0])

#apply a dropout layer to output
rnn_dropout = mx.sym.Dropout(rnn_component, p=dropout)


print("\n\t#################################\n\
       #recurrent-skip component:\n\
       #################################\n")

# connect hidden cells that are a defined time interval apart,
# because in practice very long term dependencies are not captured by LSTM/GRU
# eg if you are predicting electricity consumption you want to connect data 24hours apart
# and if your data is every 60s you make a connection between the hidden states at time t and at time t + 24h
# this connection would not be made by an LSTM since 24 is so many hidden states away

#define number of cells to skip through to get a certain time interval back from current hidden state
p = int(seasonal_period / time_interval)
print("adding skip connections for cells ", p, " intervals apart...")

#define a gated recurrent unit cell, which we can unroll into many symbols based on our desired time dependancy
skipcell_outputs = []
for i, recurrent_cell in enumerate(skiprcells):

    #unroll the rnn cell, obtaining an output and state symbol each time
    outputs, states = recurrent_cell.unroll(
        length=conv_concat_shape[1], inputs=conv_dropout, merge_outputs=False, layout="NTC")

    #for each unrolled timestep
    counter = 0
    connected_outputs = []
    for i, current_cell in enumerate(reversed(outputs)):

        #try adding a concatenated skip connection
        try:

            #get seasonal cell p steps apart
            skip_cell = outputs[i + p]

            #connect this cell to is seasonal neighbour
            cell_pair = [current_cell, skip_cell]
            concatenated_pair = mx.sym.concat(*cell_pair, dim=1)

            #prepend symbol to a list
            connected_outputs.insert(0, concatenated_pair)

            counter += 1

        except IndexError:

            #prepend symbol to a list without skip connection
            connected_outputs.insert(0, current_cell)

    selected_outputs = []
    for i, current_cell in enumerate(connected_outputs):

        t = i + 1

        #use p hidden states of Recurrent-skip component from time stamp t − p + 1 to t as outputs
        if t > conv_concat_shape[1] - p:

            selected_outputs.append(current_cell)

    #concatenate outputs
    concatenated_output = mx.sym.concat(*selected_outputs, dim=1)

    #append to list
    skipcell_outputs.append(concatenated_output)

print("\n\t", len(selected_outputs), " hidden cells used in output of shape: ",
      concatenated_pair.infer_shape(seq_data=input_feature_shape)[1][0], " after adding skip connections")

print("\n\tconcatenated output shape for each skipRNN cell type: ",
      concatenated_output.infer_shape(seq_data=input_feature_shape)[1][0])

#concatenate output from each type of recurrent cell
skiprnn_component = mx.sym.concat(*skipcell_outputs, dim=1)
print("\ncombined flattened recurrent-skip shape : ",
      skiprnn_component.infer_shape(seq_data=input_feature_shape)[1][0])

#apply a dropout layer
skiprnn_dropout = mx.sym.Dropout(skiprnn_component, p=dropout)


print("\n\t#################################\n\
       #autoregressive component:\n\
       #################################\n")

#AR component simply says the next prediction
# is the some constant times all the previous values available for that time series

auto_list = []
for i in list(range(x.shape[1])):

    #get a symbol representing data in each individual time series
    time_series = mx.sym.slice_axis(data=seq_data, axis=2, begin=i, end=i + 1)

    #pass to a fully connected layer
    fc_ts = mx.sym.FullyConnected(data=time_series, num_hidden=1)

    auto_list.append(fc_ts)

print("\neach time series shape: ", time_series.infer_shape(
    seq_data=input_feature_shape)[1][0])


#concatenate fully connected outputs
ar_output = mx.sym.concat(*auto_list, dim=1)
print("\nar component shape: ", ar_output.infer_shape(
    seq_data=input_feature_shape)[1][0])

#do not apply activation function since we want this to be linear


print("\n\t#################################\n\
       #combine AR and NN components:\n\
       #################################\n")

#combine model components
neural_components = mx.sym.concat(*[rnn_dropout, skiprnn_dropout], dim=1)

#pass to fully connected layer to map to a single value
neural_output = mx.sym.FullyConnected(
    data=neural_components, num_hidden=x.shape[1])
print("\nNN output shape : ", neural_output.infer_shape(
    seq_data=input_feature_shape)[1][0])

#sum the output from AR and deep learning
model_output = neural_output + ar_output
print("\nshape after adding autoregressive output: ",
      model_output.infer_shape(seq_data=input_feature_shape)[1][0])

#########################################
# loss function
#########################################

#compute the gradient of the L2 loss
loss_grad = mx.sym.LinearRegressionOutput(data=model_output, label=seq_label)

#set network point to back so name is interpretable
batmans_NN = loss_grad

#########################################
# create a trainable module on CPU/GPUs
#########################################

model = mx.mod.Module(symbol=batmans_NN,
                      context=context,
                      data_names=[v.name for v in train_iter.provide_data],
                      label_names=[v.name for v in train_iter.provide_label])


####################################
#define evaluation metrics to show when training
#####################################

#root relative squared error
def rse(label, pred):
    """computes the root relative squared error
    (condensed using standard deviation formula)"""

    #compute the root of the sum of the squared error
    numerator = np.sqrt(np.mean(np.square(label - pred), axis=None))
    #numerator = np.sqrt(np.sum(np.square(label - pred), axis=None))

    #compute the RMSE if we were to simply predict the average of the previous values
    denominator = np.std(label, axis=None)
    #denominator = np.sqrt(np.sum(np.square(label - np.mean(label, axis = None)), axis=None))

    return numerator / denominator


_rse = mx.metric.create(rse)

#relative absolute error


def rae(label, pred):
    """computes the relative absolute error
    (condensed using standard deviation formula)"""

    #compute the root of the sum of the squared error
    numerator = np.mean(np.abs(label - pred), axis=None)
    #numerator = np.sum(np.abs(label - pred), axis = None)

    #compute AE if we were to simply predict the average of the previous values
    denominator = np.mean(np.abs(label - np.mean(label, axis=None)), axis=None)
    #denominator = np.sum(np.abs(label - np.mean(label, axis = None)), axis=None)

    return numerator / denominator


_rae = mx.metric.create(rae)

#empirical correlation coefficient


def corr(label, pred):
    """computes the empirical correlation coefficient"""

    #compute the root of the sum of the squared error
    numerator1 = label - np.mean(label, axis=0)
    numerator2 = pred - np.mean(pred, axis=0)
    numerator = np.mean(numerator1 * numerator2, axis=0)

    #compute the root of the sum of the squared error if we were to simply predict the average of the previous values
    denominator = np.std(label, axis=0) * np.std(pred, axis=0)

    #value passed here should be 321 numbers
    return np.mean(numerator / denominator)


_corr = mx.metric.create(corr)

#create a composite metric manually
def metrics(label, pred):
    return ["RSE: ", rse(label, pred), "RAE: ", rae(label, pred), "CORR: ", corr(label, pred)]

################
# #fit the model
################

# allocate memory given the input data and label shapes
model.bind(data_shapes=train_iter.provide_data,
           label_shapes=train_iter.provide_label)

# initialize parameters by uniform random numbers
model.init_params()

# optimizer
model.init_optimizer(optimizer=optimizer,
                     optimizer_params=optimizer_params)

# train n epochs, i.e. going over the data iter one pass
for epoch in range(num_epoch):

    train_iter.reset()
    val_iter.reset()

    for batch in train_iter:
        # compute predictions
        model.forward(batch, is_train=True)
        model.backward()                                # compute gradients
        model.update()                                  # update parameters

    # compute train metrics
    pred = model.predict(train_iter).asnumpy()
    label = y_train

    print('\n', 'Epoch %d, Training %s' % (epoch, metrics(label, pred)))

    # compute test metrics
    pred = model.predict(val_iter).asnumpy()
    label = y_valid        #print('Epoch %d, Validation %s' % (epoch, metrics(label, pred)))