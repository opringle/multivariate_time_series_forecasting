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

#read in features and labels
x = np.load("../data/x.npy")
y = np.load("../data/y.npy")

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
print("\nfeature input shape: ", train_iter.provide_data, "\n",
      "\nlabel input shape: ", train_iter.provide_label, "\n")


####################################
# define neural network graph
####################################

#create placeholders to refer to when creating network graph (names are defined in data iterators)
seq_data = mx.symbol.Variable(train_iter.provide_data[0].name)
seq_label = mx.sym.Variable(train_iter.provide_label[0].name)


#create many convolutional filters to slide over the input
pooled_outputs = []
for i, filter_size in enumerate(config.filter_list):

        #convolutional layer with a kernel that slides over entire input resulting in a 1d output
        convi = mx.sym.Convolution(data=seq_data, kernel=(filter_size, config.q), num_filter=config.num_filter)

        #apply relu activation function
        acti = mx.sym.Activation(data=convi, act_type='relu')

        #append resulting symbol to a list
        pooled_outputs.append(acti)

#combine all pooled outputs (just concatenate them since they are 1d now)
concat = mx.sym.Concat(*pooled_outputs, dim=1)

#reshape for next layer (this depends on batch size, meaning train and test need same batch size)
total_filters = config.num_filter * len(config.filter_list)
h_pool = mx.sym.Reshape(data=concat, target_shape=(config.batch_size, total_filters))

#apply dropout to this layer
h_drop = mx.sym.Dropout(data=h_pool, p=config.dropout, mode='training')

#pass convolutional output into GRU recurrent layer
recurr = mx.rnn.GRUCell(num_hidden=config.recurrent_state_size)

#unroll the recurrent cell in time, obtaining a list of symbols for each time step
outputs, states = recurr.unroll(length=config.q, inputs=h_drop, merge_outputs=False, layout="NTC")


#for each unrolled timestep
step_outputs = []
for i, step_output in enumerate(outputs):
    #apply dropout to the lstm output
    drop = mx.sym.Dropout(data=step_output, p=config.dropout, mode='training')
    #add a fully connected layer with num_neurons = num_possible_tags
    fc = mx.sym.FullyConnected(data=drop, num_hidden=num_labels)
    #append symbol to a list
    step_outputs.append(fc)

#concatenate fully connected layers for each timestep (shape is now (batch_size, num_entity_labels, max_sentence_length))
concatenated_fc = mx.sym.concat(*step_outputs, dim=1)

#reshape before applying loss layer
reshaped_fc = mx.sym.Reshape(data=concatenated_fc, target_shape=(
    config.batch_size, num_labels, max_utterance_tokens))

#compute the gradient of the softmax loss, ignoring "not entity" labels
loss_grad = mx.sym.SoftmaxOutput(data=reshaped_fc, label=seq_label,
                                 use_ignore=True, ignore_label=not_entity_index, multi_output=True)

#set network point to back
lstm = loss_grad

# Visualize the network
mx.viz.plot_network(lstm, save_format='png', title="../images/network.png")

# create a trainable module on CPU/GPU
model = mx.mod.Module(symbol=lstm,
                      context=config.context,
                      data_names=[v.name for v in train_iter.provide_data],
                      label_names=[v.name for v in train_iter.provide_label])


####################################
#define evaluation metrics to show when training
#####################################

metrics = [mx.metric.create('acc'), mx.metric.create(NER_F1_score)]

eval_metrics = mx.metric.CompositeEvalMetric()
for child_metric in metrics:
    eval_metrics.add(child_metric)

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
    eval_metrics.reset()
    for batch in train_iter:
        model.forward(batch, is_train=True)       # compute predictions
        # accumulate prediction accuracy
        model.update_metric(eval_metrics, batch.label)
        model.backward()                          # compute gradients
        model.update()                            # update parameters

    print('Epoch %d, Training %s' % (epoch, eval_metrics.get()))

    eval_metrics.reset()
    for batch in val_iter:
        model.forward(batch, is_train=False)       # compute predictions
        # accumulate prediction accuracy
        model.update_metric(eval_metrics, batch.label)

    print('Epoch %d, Validation %s' % (epoch, eval_metrics.get()))


################
# save model after epochs
################

model.save_checkpoint(
    prefix='my_model',
    epoch=config.num_epoch,
    save_optimizer_states=False,
)
