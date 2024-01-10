#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 14:52:27 2023

@author:
"""

# In[ ]:

import pandas as pd
import numpy as np

# In[ ]:


"""INPUT DATAFRAME"""
data = pd.read_feather('1981_2014_matrix_temp.feather')


"""INPUT TRAINING START AND END DATE (END DATE HAS TO BE 1 DAY AHEAD OF ACTUAL END)"""
train_start = data.date[data.date == '1981-01-01 12:00:00'].index.to_list() 
train_start = train_start[0]

train_end = data.date[data.date == '2010-01-01 12:00:00'].index.to_list()              
train_end = train_end[0]

"""List of column names for feature importance"""
a = data.columns[2:33]


"""INPUT TESTING START AND END DATE"""
test_start = data.date[data.date == '2010-01-01 12:00:00'].index.to_list()                
test_start = test_start[0]



test_end = data.date[data.date == '2013-01-01 12:00:00'].index.to_list()                 
test_end = test_end[0]


"""DEFINE TESTING AND TRAINING SETS. EMBEDDING VARIABLES AT THE END
IF UNSURE OF ORDER USE data.columns[] to check order"""



test_features_raw = data.iloc[test_start:test_end,2:33].to_numpy()
test_targets = data.iloc[test_start:test_end,1].to_numpy()
test_IDs = data.iloc[test_start:test_end,34].to_numpy()
lead_test=data.iloc[test_start:test_end,33].to_numpy()


train_features_raw = data.iloc[:train_end,2:33].to_numpy()      #all other features being considered
train_targets = data.iloc[:train_end,1].to_numpy()              #observed data
train_IDs = data.iloc[:train_end,34].to_numpy()                 #station numbers
lead=data.iloc[:train_end,33].to_numpy()





#%%NORMALIZATION

"""SOME VARIABLES WILL NOT BE NORMALIZED AGAIN. WE SAVE THEM HERE AND ADD THEM BACK
TO THE DATAFRAME AFTER NORMALIZATION"""

ao = data.ao
nao = data.nao
doy = data.doy
sin_doy = data['sin-doy']



# normalize data

def normalize(data, method=None, shift=None, scale=None):
    result = np.zeros(data.shape)
    if method == "MAX":
        scale = np.max(data, axis=0)
        shift = np.zeros(scale.shape)
    for index in range(len(data[0])):
        result[:,index] = (data[:,index] - shift[index]) / scale[index]
    return result, shift, scale

train_features, train_shift, train_scale = normalize(train_features_raw[:,:], method="MAX")

test_features = normalize(test_features_raw[:,:], shift=train_shift, scale=train_scale)[0]

#adding the original nao,ao and doy back in to the train/test sets since they are already normalized
#value in the brackets determined by their position in the train/test features variable
train_features[:,18] = nao.iloc[train_start:train_end]
train_features[:,19] = ao.iloc[train_start:train_end]
train_features[:,22] = doy.iloc[train_start:train_end]
train_features[:,23] = sin_doy.iloc[train_start:train_end]


test_features[:,18] = nao.iloc[test_start:test_end]
test_features[:,19] = ao.iloc[test_start:test_end]
test_features[:,22] = doy.iloc[test_start:test_end]
test_features[:,23] =  sin_doy.iloc[test_start:test_end]



del data,nao,ao,doy,train_features_raw,test_features_raw


# In[ ]:


# helper functions for NN models

import tensorflow as tf
import keras
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Input, Dense, Embedding, Flatten, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam,SGD

def build_EMOS_network_keras(compile=False, optimizer='sgd', lr=0.1):
    """Build (and maybe compile) EMOS network in keras.

    Args:
        compile: If true, compile model
        optimizer: String of keras optimizer
        lr: learning rate

    Returns:
        model: Keras model
    """
    mean_in = Input(shape=(1,))
    std_in = Input(shape=(1,))
    mean_out = Dense(1, activation='linear')(mean_in)
    std_out = Dense(1, activation='linear')(std_in)
    x = keras.layers.concatenate([mean_out, std_out], axis=1)
    model = Model(inputs=[mean_in, std_in], outputs=x)

    if compile:
        opt = SGD(learning_rate=lr)
        model.compile(optimizer=opt, loss=crps_cost_function)
    return model


def crps_cost_function(y_true, y_pred, theano=False):
    """Compute the CRPS cost function for a normal distribution defined by
    the mean and standard deviation.

    Code inspired by Kai Polsterer (HITS).

    Args:
        y_true: True values
        y_pred: Tensor containing predictions: [mean, std]
        theano: Set to true if using this with pure theano.

    Returns:
        mean_crps: Scalar with mean CRPS over batch
    """

    # Split input
    mu = y_pred[:, 0]
    sigma = y_pred[:, 1]
    # Ugly workaround for different tensor allocation in keras and theano
    if not theano:
        y_true = y_true[:, 0]   # Need to also get rid of axis 1 to match!

    # To stop sigma from becoming negative we first have to 
    # convert it the the variance and then take the square
    # root again. 
    var = sigma ** 2
    # The following three variables are just for convenience
    loc = (y_true - mu) / tf.sqrt(var)
    phi = 1.0 / tf.sqrt(2.0 * np.pi) * tf.exp(-loc ** 2 / 2.0)
    Phi = 0.5 * (1.0 + tf.math.erf(loc / tf.sqrt(2.0)))
    # First we will compute the crps for each input/target pair
    crps =  tf.sqrt(var) * (loc * (2. * Phi - 1.) + 2 * phi - 1. / tf.sqrt(np.pi))
    # Then we take the mean. The cost is now a scalar
    return tf.math.reduce_mean(crps)

def build_hidden_model(n_features, n_outputs, hidden_nodes, compile=False,
                       optimizer='adam', lr=0.01, loss=crps_cost_function,
                       activation='relu'):
    """Build (and compile) a neural net with hidden layers
    Args:
        n_features: Number of features
        n_outputs: Number of outputs
        hidden_nodes: int or list of hidden nodes
        compile: If true, compile model
        optimizer: Name of optimizer
        lr: learning rate
        loss: loss function
    Returns:
        model: Keras model
    """
    if type(hidden_nodes) is not list:
        hidden_nodes = [hidden_nodes]
    inp = Input(shape=(n_features,))
    x = Dense(hidden_nodes[0], activation=activation)(inp)
    if len(hidden_nodes) > 1:
        for h in hidden_nodes[1:]:
            x = Dense(h, activation=activation)(x)
    x = Dense(n_outputs, activation='linear')(x)
    model = Model(inputs=inp, outputs=x)

    if compile:
        opt = Adam(learning_rate=lr)
        model.compile(optimizer=opt, loss=loss)
    return model

def build_emb_model(n_features, n_outputs, hidden_nodes, emb_size, max_id, 
                    compile=False, lr=0.002,
                    loss=crps_cost_function,
                    activation='relu', reg=None):
    """
    Args:
        n_features: Number of features
        n_outputs: Number of outputs
        hidden_nodes: int or list of hidden nodes
        emb_size: Embedding size
        max_id: Max embedding ID
        compile: If true, compile model
        lr: learning rate
        loss: loss function
        activation: Activation function for hidden layer

    Returns:
        model: Keras model
    """
    if type(hidden_nodes) is not list:
        hidden_nodes = [hidden_nodes]

    features_in = Input(shape=(n_features,))
    #Layer to be used as an entry point into a Network (a graph of layers).
    
    id_in = Input(shape=(1,))
    lead_time=Input(shape=(1,))
    emb = Embedding(max_id + 1, emb_size)(id_in)
    emb1=Embedding(max_id +1, emb_size)(lead_time)
    #Turns positive integers (indexes) into dense vectors of fixed size
    
    emb = Flatten()(emb)
    emb1=Flatten()(emb1)
    #flattens input to 1d array
    
    x = Concatenate()([features_in, emb,emb1])
    #Layer that concatenates a list of inputs
    
    for h in hidden_nodes:
        x = Dense(h, activation=activation)(x)
        
    
    # x = Dense(6, activation='relu',kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=0.01))(x)
    x = Dense(n_outputs, activation='linear')(x)
    model = Model(inputs=[features_in, id_in,lead_time], outputs=x)

    if compile:
        opt = Adam(learning_rate=lr)
        model.compile(optimizer=opt, loss=loss)
    return model

def build_fc_model(n_features, n_outputs, compile=False, optimizer='adam',
                   lr=0.02, loss=crps_cost_function):
    """Build (and compile) fully connected linear network.
    Args:
        n_features: Number of features
        n_outputs: Number of outputs
        compile: If true, compile model
        optimizer: Name of optimizer
        lr: learning rate
        loss: loss function
    Returns:
        model: Keras model
    """

    inp = Input(shape=(n_features,))
    x = Dense(n_outputs, activation='linear')(inp)
    model = Model(inputs=inp, outputs=x)

    if compile:
        opt = Adam(learning_rate=lr)
        model.compile(optimizer=opt, loss=loss)
    return model

#%%NN-aux-emb

from tensorflow.keras.backend import clear_session

# training multiple models in a loop

emb_size = 3
max_id = int(tf.math.reduce_max([train_IDs.max(), test_IDs.max()]))     #max station ID across training or testing sets
n_features = train_features.shape[1]                                    #1-D integer tensor representing the shape of input
n_outputs = 2

# nreps = 10
trn_scores = []
test_scores = []
preds = []

clear_session()

features_in = Input(shape=(n_features,))

#embeddings
id_in = Input(shape=(1,))
lead_time=Input(shape=(1,))

emb = Embedding(max_id + 1, emb_size)(id_in)

emb1=Embedding(max_id +1, emb_size)(lead_time)


emb = Flatten()(emb)
emb1=Flatten()(emb1)
#flattens input to 1d array

#can remove embedding from here if needed
x = Concatenate()([features_in, emb,emb1])
#Layer that concatenates the list of inputs

"""Additional Dense layers can be added or removed here as seen fit"""

x = Dense(14, activation='elu')(x)

x = Dense(6, activation='elu')(x)

x = Dense(4, activation='elu')(x)

x = Dense(n_outputs, activation='linear')(x)    #output layer

nn_aux_emb = Model(inputs=[features_in, id_in,lead_time], outputs=x)

nn_aux_emb.summary()
opt = Adam(learning_rate=0.002)

nn_aux_emb.compile(optimizer=opt, loss=crps_cost_function)  

nn_aux_emb.fit([train_features, train_IDs,lead], train_targets, epochs=5, batch_size=16384*2,verbose=1)   
"""Fit the model using the training data"""

trn_scores.append(nn_aux_emb.evaluate([train_features, train_IDs,lead], train_targets, 16384*2, verbose=1))
test_scores.append(nn_aux_emb.evaluate([test_features, test_IDs,lead_test], test_targets, 4096*2, verbose=1))
"""Evaluating the model on training and testing data sets"""
preds.append(nn_aux_emb.predict([test_features, test_IDs,lead_test], 4096*2 , verbose=0))
"""Predicting values for test period"""

"""Saving array of predictions for further computations"""
# arr = np.asarray(preds)
# np.save('preds_01_05_upd.npy',arr)



#%%FIMP SAVE
ref_score = nn_aux_emb.evaluate([test_features,test_IDs,lead_test], test_targets, 16384, verbose=0)


def eval_shuf(m, idx, emb=False):
    x_shuf = test_features.copy()
    x_shuf[:, idx] = np.random.permutation(x_shuf[:, idx])
    x = [x_shuf, test_IDs, lead_test] if emb else x_shuf
    return m.evaluate(x, test_targets, 16384, 0)

def perm_imp_emb(m, ref):
    scores = [eval_shuf(m, i, True) for i in range(len(a))]
    ids_shuf = np.random.permutation(test_IDs)
    lead_shuf = np.random.permutation(lead_test)
    scores += [m.evaluate([test_features,ids_shuf,lead_shuf],test_targets, 16384, 0)]
    fimp = np.array(scores) - ref
    df = pd.DataFrame(columns=['Feature', 'Importance'])
    a2 = a.union(['Embedding'],sort=False)
    df['Feature'] = a2; df['Importance'] = fimp
    return df

a = data.columns[2:33]
fimp_fc_aux_emb = perm_imp_emb(nn_aux_emb, ref_score)
fimp_fc_aux_emb.to_feather('nn_aux_emb_fimp_upd.feather')
import matplotlib.pyplot as plt
import seaborn as sns
fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(data=fimp_fc_aux_emb, y='Importance', x='Feature', ax=ax)
plt.xticks(rotation=90);
# plt.savefig("3M_lead/feature_imp_nn_2emb.png",dpi=800,bbox_inches = 'tight')
