import gym
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import interactive
interactive(True)
import gym_trading

env = gym.make('trading-v0')
#env.time_cost_bps = 0 # 

env = env.unwrapped



import tensorflow as tf
import policy_gradient


# create the tf session
sess = tf.InteractiveSession()

# Input
num_actions=10  # same as # of stocks
Variables=3
obs_dim=num_actions*Variables
NumOfLayers=2
architecture = "LSTM" # right now, valid inputs are LSTM and FFNN
LR="GD"
actFunc="relu"
regulizer="l2"
regulizerScale=0.0001
#avgfilename="/home/s1793158/mlp3/FILE_Name.p"

actFuncs=["relu","elu","lrelu","selu","sigmoid"]
name=["RNN_GD_relu_l2","RNN_GD_elu_l2","RNN_GD_lrelu_l2","RNN_GD_selu_l2","RNN_GD_sig_l2"]
   # from tensorflow.python.framework import ops
    #ops.reset.default_graph()

for i in range(5):
    tf.reset_default_graph()
    sess = tf.InteractiveSession()
    
    avgfilename="/home/s1793158/mlp3/Saving/"+name[i]+".p"
    Modelfilename="/home/s1793158/mlp3/mlp3/SavedModels/"+name[i]
    
    
    pg = policy_gradient.PolicyGradient(sess, obs_dim=obs_dim, 
                                        num_actions=num_actions,
                                        NumOfLayers=NumOfLayers, 
                                        LR=LR,
                                        architecture=architecture,
                                        actFunc=actFuncs[i],
                                        learning_rate=1e-4,
                                        regulizer =regulizer,
                                        regulizerScale=regulizerScale,
                                        avgfilename=avgfilename,
                                        Modelfilename=Modelfilename
                                       )

    # and now let's train it and evaluate its progress.  NB: this could take some time...
    direc="aa"
    load_model=False
    #grads, grads_clipped = pg.get_grads_and_clipping()
    '''with tf.Session() as sess:
    sess.run(init)  # actually running the initialization op
    _grads_clipped, _grads = sess.run(
                        [grads_clipped, grads],
                        feed_dict={pg.X: epX, pg._tf_epr: epr, pg._tf_y: epy, pg._tf_x: epx})'''
    df,sf = pg.train_model(env, episodes=5, log_freq=100, load_model=False,model_dir = direc)#, load_model=True)