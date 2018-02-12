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
NumOfLayers=3
architecture = "LSTM" # right now, valid inputs are LSTM and FFNN
LR="Adam"
actFunc="relu"
regulizer="l2"
regulizerScale=0.01
avgfilename="randomfilename.p"


pg = policy_gradient.PolicyGradient(sess, obs_dim=obs_dim, 
                                    num_actions=num_actions,
                                    NumOfLayers=NumOfLayers, 
                                    LR=LR,
                                    architecture=architecture,
                                    actFunc=actFunc,
                                    learning_rate=1e-2,
                                    regulizer =regulizer,
                                    regulizerScale=regulizerScale,
                                    avgfilename=avgfilename
                                   )

# and now let's train it and evaluate its progress.  NB: this could take some time...
model_dir="aa"
load_model=False
#grads, grads_clipped = pg.get_grads_and_clipping()
'''with tf.Session() as sess:
    sess.run(init)  # actually running the initialization op
    _grads_clipped, _grads = sess.run(
                        [grads_clipped, grads],
                        feed_dict={pg.X: epX, pg._tf_epr: epr, pg._tf_y: epy, pg._tf_x: epx})'''
df,sf = pg.train_model(env, episodes=5, log_freq=1, load_model=False,model_dir = direc)#, load_model=True)
