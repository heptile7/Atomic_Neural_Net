#!/usr/bin/env python
# coding: utf-8

# This notebooks covers more details on tweaking and optimizing the training process.



#Tested in tensorflow=1.13.1
#Requires PiNN(see below), matplotlib,python 3.6 and above
#For PiNN, https://teoroo-pinn.readthedocs.io/en/latest 
import os, warnings
import tensorflow as tf

import matplotlib.pyplot as plt

from glob import glob
from pinn.io import load_qm9, sparse_batch
from pinn.networks import pinet,bpnn
from pinn.utils import get_atomic_dress
from pinn.models import potential_model,dipole_model
import random

os.environ['CUDA_VISIBLE_DEVICES'] = ''
index_warning = 'Converting sparse IndexedSlices'
warnings.filterwarnings('ignore', index_warning)


# Optimizing the pipeline
# Caching
# Caching stores the decoded dataset in the memory.



#Dataset Caching for pre-processing
#For the purpose of testing, we use only 10 samples from QM9
datalist = (glob(r'C:\Users\phy\Desktop\hema\Materials_Design/Dataset/QM9/dsgdb9nsd/*.xyz'))

filelist=[]
r=[]
r=random.sample(range(len(datalist)),10) #Choosing 10 random samples of molecules

for j in range(len(r)):
    filelist.append(datalist[r[j]]) #appends the randomly selected molecular data in a list
len(filelist)
dataset = lambda: load_qm9(filelist, split=1,label_map={'d_data':'mu'}) #Selects only the d_data i.e., the Dipole Moment


d = dataset().cache().repeat().apply(sparse_batch(10))
tensors = d.make_one_shot_iterator().get_next() #Iterators duplicate data from given data for creating datasets of 'similar' pattern
with tf.Session() as sess:
    for i in range(10):
        sess.run(tensors) # "Warm up" the graph
    get_ipython().run_line_magic('timeit', 'sess.run(tensors)')


# This speed indicates the IO limit of our current setting.
# Now let's cache the dataset to the memory.

d = dataset().cache().repeat().apply(sparse_batch(10)) #These dataset iterators help create multiple datas with similar patterns for easier model building
tensors = d.make_one_shot_iterator().get_next()
with tf.Session() as sess:
    for i in range(10):
        sess.run(tensors) # "Warm up" the graph, dataset is cached here
    get_ipython().run_line_magic('timeit', 'sess.run(tensors)')


# Preprocessing
# You might also see a notable difference in the performance with and without preprocessing. This is especially helpful when you are training with GPUs.

d = dataset().cache().repeat().apply(sparse_batch(10)) #Pre-Processing is done for faster training of models as the data is cached before building the model
tensors = d.make_one_shot_iterator().get_next()
output = pinet(tensors)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(10):
        sess.run(output)
    get_ipython().run_line_magic('timeit', 'sess.run(output)')

tf.reset_default_graph()

pre_fn = lambda tensors: pinet(tensors, preprocess=True)
d = dataset().cache().repeat().apply(sparse_batch(10)).map(pre_fn, 8)
tensors = d.make_one_shot_iterator().get_next()
output = pinet(tensors)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(10):
        sess.run(output)
    get_ipython().run_line_magic('timeit', 'sess.run(output)')


# You can even cache the preprocessed data.

tf.reset_default_graph()

pre_fn = lambda tensors: pinet(tensors, preprocess=True)
d = dataset().apply(sparse_batch(10)).map(pre_fn,8).cache().repeat()
tensors = d.make_one_shot_iterator().get_next()
output = pinet(tensors)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(10):
        sess.run(output)
    get_ipython().run_line_magic('timeit', 'sess.run(output)')

N_molecules=100 #Choice of the number of molecular samples to build model using

datalist = (glob(r'C:\Users\phy\Desktop\hema\Materials_Design/Dataset/QM9/dsgdb9nsd/*.xyz')) #dataset location

filelist=[]
r=[]
r=random.sample(range(len(datalist)),N_molecules) #Random Choice of N_Molecules number of molecular data 

for j in range(len(r)):
    filelist.append(datalist[r[j]])
len(filelist)
dataset = lambda: load_qm9(filelist, split={'train':8, 'test':2},label_map={'d_data':'mu'}) #d_data selects only dipole moment from the filelist of QM9 dataset
#dataset is split into training and testing datasets

# Training with the optimized pipeline

node_no=64 #no.of nodes in a single layer
GC_iterations=5 #GC block iteration number for PiNet 
#parameters and hyper-parameters of the model Architecture of choice
params = {'model_dir': 'QM9_dipole_l2_x{}_{}_depth_{}'.format(node_no,N_molecules,GC_iterations), #Model is saved in this folder location
          'network': 'pinet', #Choice of ANN
          'network_params': {
              'atom_types':[1, 6, 7, 8, 9],
              'pi_nodes':[node_no]*10,
              'ii_nodes':[node_no,node_no,node_no,node_no],
              'pp_nodes':[node_no,node_no,node_no,node_no],
              'act':'tanh',
              'en_nodes':[node_no,1],
              'depth':GC_iterations,
              'rc':4.5,
              'basis_type':'gaussian',
              'n_basis':10
              },
          'model_params': {
              'learning_rate': 1e-4, #learning rate
              #'use_decay':True, #Decay is obsolete for smaller training steps
              #'decay_step':100000, 
              #'decay_rate':0.994,           
              #'use_norm_clip':True,
              #'use_d_per_atom':True,
              #'use_l2':True,
              #'use_d_per_sqrt':True
              }}
# The logging behavior of estimator can be controlled here(Saving model_data after certain time)
config = tf.estimator.RunConfig(log_step_count_steps=1000,save_checkpoints_steps=1000) #Can be modified to use in GPUs. Check tensorflow documentation for command
#Set mini-batch size
if N_molecules<=100:
    batch_size=N_molecules
if N_molecules>100:
    batch_size=100
batch_size=10
#Preprocessing the datasets
pre_fn = lambda tensors: pinet(tensors, preprocess=True, **params['network_params'])
train = lambda: dataset()['train'].cache().repeat().shuffle(1000).apply(sparse_batch(batch_size)).map(pre_fn, 8)
test = lambda: dataset()['test'].cache().repeat().apply(sparse_batch(batch_size)).map(pre_fn, 8)

# Running specs
train_spec = tf.estimator.TrainSpec(input_fn=train, max_steps=1e4) #Configuration of Training dataset and number of training steps
eval_spec = tf.estimator.EvalSpec(input_fn=test, steps=100,throttle_secs=90,start_delay_secs=45) #Triggering the testing dataset for evaluation and its accuracy

model = dipole_model(params, config=config) #Type of model to build
tf.estimator.train_and_evaluate(model, train_spec, eval_spec) #Start Build Model


# Monitoring
# It's recommended to monitor the training with Tensorboard instead of the stdout here.  

# tensorboard --logdir /model #Typed into the cmd prompt gives the tensorboard for visualizing and monitoring the model


# Parallelization with tf.Estimator

# The estimator api makes it extremely easy to train on multiple GPUs.
