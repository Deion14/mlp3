# https://deeplearningcourses.com/c/deep-reinforcement-learning-in-python
# https://www.udemy.com/deep-reinforcement-learning-in-python
from __future__ import print_function, division
from builtins import range

# Note: you may need to update your version of future
# sudo pip install -U future

import gym
import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from gym import wrappers
from datetime import datetime
from math import isnan
import matplotlib as mpl
from matplotlib import interactive
interactive(True)
import gym_trading
import time

#import policy_gradient

# so you can test different architectures
class HiddenLayer:
  def __init__(self, M1, M2, f=tf.nn.tanh, use_bias=True, zeros=False):
    if zeros:
      W = np.zeros((M1, M2), dtype=np.float32)
    else:
      W = tf.random_normal(shape=(M1, M2)) * np.sqrt(2. / M1, dtype=np.float32)
    self.W = tf.Variable(W)

    self.use_bias = use_bias
    if use_bias:
      self.b = tf.Variable(np.zeros(M2).astype(np.float32))

    self.f = f

  def forward(self, X):
    if self.use_bias:
      a = tf.matmul(X, self.W) + self.b
    else:
      a = tf.matmul(X, self.W)
    return self.f(a)


# approximates pi(a | s)
class PolicyModel:
  def __init__(self, D, A, 
               NumOfLayers,
               Num_Of_variables,
               LR,
               architecture,
               actFunc,
               learning_rate,
               regulizer,
               regulizerScale,
               num_hiddenRNN,
               DropoutMemoryStates,
               DropoutVariational_recurrent,
               output_keep_prob,
               state_keep_prob):
      
      
    self.D = D
    self.A = A
    self.T = 63
    
    # inputs and targets
    self.X = tf.placeholder(tf.float32, shape=(None, 63, self.D), name='X_for_policy')
    self.actions = tf.placeholder(tf.float32, shape=(None,self.A), name='actions')
    self.advantages = tf.placeholder(tf.float32, shape=(None,1), name='advantages')
    
 #   self.mean_layer = HiddenLayer(10, 10, lambda x: x, use_bias=False, zeros=True)
 #   self.stdv_layer = HiddenLayer(10, 10, tf.nn.softplus, use_bias=False, zeros=False)
    
    # get final hidden layer
    #with tf.variable_scope('policy_rnn', reuse=tf.AUTO_REUSE): 
     #   p_val, _  = tf.nn.dynamic_rnn(policy_cell, self.X, dtype=tf.float32)
    

    self.num_hiddenRNN=num_hiddenRNN
    self.output_keep_prob=output_keep_prob  
    self.state_keep_prob=state_keep_prob
    self.DropoutVariational_recurrent=DropoutVariational_recurrent
    self._variables=Num_Of_variables
    self.NumofLayers=NumOfLayers
    
    if actFunc=="sigmoid":
        self.actFunc=tf.nn.sigmoid
    elif  actFunc=="relu":
            self.actFunc=tf.nn.relu
    elif actFunc=="lrelu":
            self.actFunc= tf.nn.leaky_relu
    elif actFunc=="elu":
            self.actFunc= tf.nn.elu   
    elif actFunc=="selu":
            self.actFunc= tf.nn.selu                   
    else:
        assert("wrong acttivation function")



    self.architecture=architecture
    self.DropoutMemoryStates=DropoutMemoryStates
    self.learningRule = LR
    self.learning_rate = learning_rate
    
    
    def dropout_state_filter_visitors(state): 
            ''' Dropout of memory cells Not literature based on tensorflow code'''
            
            if isinstance(state, tf.contrib.rnn.LSTMStateTuple): # Never perform dropout on the c state.
                return tf.contrib.rnn.LSTMStateTuple(c=True, h=True) 
            elif isinstance(state, tf.TensorArray): 
                return False 
            return True

    
    

    def get_a_cell(num_hidden,i):
            ''' Function for GRU, RNN, LSTM'''

            cell_type=self.architecture

            if cell_type == 'GRU':
                cell = tf.nn.rnn_cell.GRUCell(self.num_hiddenRNN,activation=self.actFunc)
            elif cell_type == 'LSTM':
                cell = tf.nn.rnn_cell.LSTMCell(self.num_hiddenRNN, state_is_tuple=True,activation=self.actFunc)
            elif cell_type == 'RNN':
                cell = tf.nn.rnn_cell.BasicRNNCell(self.num_hiddenRNN,activation=self.actFunc)

#                    pdb.set_trace()

            drop = tf.nn.rnn_cell.DropoutWrapper(cell, 
                                                 input_keep_prob=1,
                                                 output_keep_prob=self.output_keep_prob,
                                                 state_keep_prob=self.state_keep_prob,
                                                 variational_recurrent=self.DropoutVariational_recurrent,
                                                 input_size=self.A*self._variables if i==0 else tf.TensorShape(num_hidden), 
                                                 dtype=tf.float32,
                                                 seed=None,
                                                 dropout_state_filter_visitor=dropout_state_filter_visitors if self.DropoutMemoryStates==True else None                                                               )

            return drop

    ''' Create Stacked Model '''    
    with tf.name_scope('actor_model'):
          cell = tf.nn.rnn_cell.MultiRNNCell(
          [get_a_cell(self.num_hiddenRNN, i) for i in range(self.NumofLayers)])
    ''' Make it runable '''
    with tf.variable_scope('actor', initializer=tf.contrib.layers.xavier_initializer()):
        h, _ =  tf.nn.dynamic_rnn(cell, self.X, dtype=tf.float32) 
        
    
    p_val = tf.reshape(h,[-1,self.num_hiddenRNN*63])

    init = tf.contrib.layers.xavier_initializer(uniform=False, dtype=tf.float32)
    output=tf.contrib.layers.fully_connected(p_val,#tf.contrib.layers.flatten(h),
                                         self.A,
                                         activation_fn=None,
                                         normalizer_fn=None,
                                         normalizer_params=None,
                                         weights_initializer=init,
                                         weights_regularizer=None,
                                         biases_initializer=tf.zeros_initializer(),
                                         biases_regularizer=None,
                                         reuse=None,
                                         variables_collections=None,
                                         outputs_collections=None,
                                         trainable=True,scope=None)
    
    sign=tf.sign(output)
    absLogP=tf.abs(output)
    P = tf.multiply(sign,tf.nn.softmax(absLogP))

    self.predict_op = P    
        
    # calculate output and cost
#    mean = self.mean_layer.forward(P)
#    stdv = self.stdv_layer.forward(P) + 1e-5 # smoothing
#
#    norm = tf.distributions.Normal(mean, stdv)
#    log_probs = norm.log_prob(self.actions)
    
    action_ratio = tf.div(P,self.actions)
    action_ratio = tf.clip_by_value(action_ratio, -1.2, 1.2)
    
    cost = -tf.reduce_sum(self.advantages * action_ratio)
    self.cost = cost
    
    # Different Learning Rules
    if self.learningRule=="RMSProp":
        self.train_op = tf.train.RMSPropOptimizer(self.learning_rate).minimize(cost)
    elif self.learningRule=="Adam":
        self.train_op = tf.train.AdamOptimizer(self.learning_rate,
                                           beta1=0.9,
                                           beta2=0.999).minimize(cost)
    elif self.learningRule=="Mom":
        self.train_op = tf.train.MomentumOptimizer(self.learning_rate,
                                            momentum=0.8,
                                            use_locking=False,
                                            use_nesterov=True).minimize(cost)
    elif self.learningRule=="GD":
        self.train_op = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(cost)

  def set_session(self, session):
    self.session = session

  def partial_fit(self, X, actions, advantages):
    
    #X = np.reshape(X, (-1, self.T, self.D))
    
    actions = np.reshape(actions, (-1, self.A))
    advantages = np.reshape(advantages, (-1, 1))
    self.session.run(
      self.train_op,
      feed_dict={
        self.X: X,
        self.actions: actions,
        self.advantages: advantages,
      }
    )

  def predict(self, X):
    
    #X = self.ft.transform(X)
    return self.session.run(self.predict_op, feed_dict={self.X: X})

  def sample_action(self, X):
    #X = np.reshape(X, (-1, self.T, self.D))
    p = self.predict(X)
    return p


# approximates V(s)
class ValueModel:
  def __init__(self, D, A, 
               NumOfLayers,
               Num_Of_variables,
               LR,
               architecture,
               actFunc,
               learning_rate,
               regulizer,
               regulizerScale,
               num_hiddenRNN,
               DropoutMemoryStates,
               DropoutVariational_recurrent,
               output_keep_prob,
               state_keep_prob):
    
    self.D = D
    self.A = A
    self.T = 63
    self.T = 63
    self.costs = []

#    # inputs and targets
    self.X = tf.placeholder(tf.float32, shape=(None, 63, self.D), name='X_for_value')
    self.Y = tf.placeholder(tf.float32, shape=(None,1), name='Y')     
        
        
    self.num_hiddenRNN=num_hiddenRNN    
    self.output_keep_prob=output_keep_prob  
    self.state_keep_prob=state_keep_prob
    self.DropoutVariational_recurrent=DropoutVariational_recurrent
    self._variables=Num_Of_variables
    self.NumofLayers=NumOfLayers
    

    if actFunc=="sigmoid":
        self.actFunc=tf.nn.sigmoid
    elif  actFunc=="relu":
            self.actFunc=tf.nn.relu
    elif actFunc=="lrelu":
            self.actFunc= tf.nn.leaky_relu
    elif actFunc=="elu":
            self.actFunc= tf.nn.elu   
    elif actFunc=="selu":
            self.actFunc= tf.nn.selu                   
    else:
        assert("wrong acttivation function")



    self.architecture=architecture
    self.DropoutMemoryStates=DropoutMemoryStates
    self.learningRule = LR
    self.learning_rate = learning_rate
    
    
    def dropout_state_filter_visitors(state): 
            ''' Dropout of memory cells Not literature based on tensorflow code'''
            
            if isinstance(state, tf.contrib.rnn.LSTMStateTuple): # Never perform dropout on the c state.
                return tf.contrib.rnn.LSTMStateTuple(c=True, h=True) 
            elif isinstance(state, tf.TensorArray): 
                return False 
            return True

    
    

    def get_a_cell(num_hidden,i):
            ''' Function for GRU, RNN, LSTM'''

            cell_type=self.architecture

            if cell_type == 'GRU':
                cell = tf.nn.rnn_cell.GRUCell(self.num_hiddenRNN,activation=self.actFunc)
            elif cell_type == 'LSTM':
                cell = tf.nn.rnn_cell.LSTMCell(self.num_hiddenRNN, state_is_tuple=True,activation=self.actFunc)
            elif cell_type == 'RNN':
                cell = tf.nn.rnn_cell.BasicRNNCell(self.num_hiddenRNN,activation=self.actFunc)

#                    pdb.set_trace()

            drop = tf.nn.rnn_cell.DropoutWrapper(cell, 
                                                 input_keep_prob=1,
                                                 output_keep_prob=self.output_keep_prob,
                                                 state_keep_prob=self.state_keep_prob,
                                                 variational_recurrent=self.DropoutVariational_recurrent,
                                                 input_size=self.A*self._variables if i==0 else tf.TensorShape(num_hidden), 
                                                 dtype=tf.float32,
                                                 seed=None,
                                                 dropout_state_filter_visitor=dropout_state_filter_visitors if self.DropoutMemoryStates==True else None                                                               )

            return drop

    ''' Create Stacked Model '''    
    with tf.name_scope('critic_model'):
          cell = tf.nn.rnn_cell.MultiRNNCell(
          [get_a_cell(self.num_hiddenRNN, i) for i in range(self.NumofLayers)])
    ''' Make it runable '''
    with tf.variable_scope('critic', initializer=tf.contrib.layers.xavier_initializer()):
        h, _ =  tf.nn.dynamic_rnn(cell, self.X, dtype=tf.float32) 
                 
    v_val = tf.reshape(h,[-1,self.num_hiddenRNN*63])


    init = tf.contrib.layers.xavier_initializer(uniform=False, dtype=tf.float32)
    Y_hat = tf.contrib.layers.fully_connected(v_val,#tf.contrib.layers.flatten(h),
                                         1,
                                         activation_fn=None,
                                         normalizer_fn=None,
                                         normalizer_params=None,
                                         weights_initializer=init,
                                         weights_regularizer=None,
                                         biases_initializer=tf.zeros_initializer(),
                                         biases_regularizer=None,
                                         reuse=None,
                                         variables_collections=None,
                                         outputs_collections=None,
                                         trainable=True,scope=None)        
        
    self.predict_op = Y_hat

    cost = tf.reduce_sum(tf.square(self.Y - Y_hat))
    self.cost = cost
    
    # Different Learning Rules
    if self.learningRule=="RMSProp":
        self.train_op = tf.train.RMSPropOptimizer(self.learning_rate).minimize(cost)
    elif self.learningRule=="Adam":
        self.train_op = tf.train.AdamOptimizer(self.learning_rate,
                                           beta1=0.9,
                                           beta2=0.999).minimize(cost)
    elif self.learningRule=="Mom":
        self.train_op = tf.train.MomentumOptimizer(self.learning_rate,
                                            momentum=0.8,
                                            use_locking=False,
                                            use_nesterov=True).minimize(cost)
    elif self.learningRule=="GD":
        self.train_op = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(cost)

  def set_session(self, session):
    self.session = session

  def partial_fit(self, X, Y):

    #X = np.reshape(X, (-1, self.T, self.D))
    Y = np.reshape(Y, (-1, 1))
    self.session.run(self.train_op, feed_dict={self.X: X, self.Y: Y})
    cost = self.session.run(self.cost, feed_dict={self.X: X, self.Y: Y})
    self.costs.append(cost)

  def predict(self, X):

    #X = np.reshape(X, (-1, self.T, self.D))
    return self.session.run(self.predict_op, feed_dict={self.X: X})


def play_one_td(envs, pmodel, vmodel, gamma):
  
  env, testing_env = envs[0], envs[1]  
    
  observation,_ = env.reset()
  testing_obs,_ = testing_env.reset()
  
  
  action = pmodel.sample_action(observation)
  prev_observation = observation
  observation, reward, done, sort , info, _ = env.step(action)
  reward = np.reshape(reward, (-1,1))
  

  test_action = pmodel.sample_action(testing_obs)
  _, _, _, test_sort , test_info, _ = testing_env.step(test_action)
  
  # update the models
  V_next = vmodel.predict(observation)
  G = reward + gamma*V_next
  advantage = G - vmodel.predict(prev_observation)
  
  pmodel.partial_fit(prev_observation, action, advantage)
  vmodel.partial_fit(prev_observation, G)
    
  #return np.array(total_rewards[-150:]).mean(), iters
  return sort, info['nominal_reward'], test_sort, test_info['nominal_reward']
  

def training():
    
  env_trading = gym.make('trading-v0')
  env_trading = env_trading.unwrapped
  
  env_testing = gym.make('testing-v0')
  env_testing = env_testing.unwrapped

  
  name=["2nd_LSTM_Adam_10e4_1stack_100",
        "2nd_LSTM_Adam_10e4_2stack_24",
        "2nd_LSTM_Adam_10e4_2stack_100",
        "2nd_LSTM_Adam_10e5_1stack_100",
        "2nd_LSTM_Adam_10e5_2stack_24",
        "2nd_LSTM_Adam_10e5_2stack_100"]

  name = ["3rd_LSTM_Adam_10e4_2stack_100_drop"]
  
  actFuncs="lrelu"
  NumOfHiddLayers = [2]    
  output_keep_prob = .6
  state_keep_prob = .6
  DropoutVariational_recurrent = False
  Num_Of_variables = 3
  num_hiddenRNN = [100]
  architecture = 'LSTM'
  DropoutMemoryStates = False
  LR = 'Adam'
  learning_rate = [1e-4]
  regulizer="l2"
  regulizerScale=0.0001
    
  D,A = 30, 10
  
  for i in range(len(name)):      
      tf.reset_default_graph()
      #D = ft.dimensions
 
      pmodel = PolicyModel(D, A, 
                           NumOfLayers=NumOfHiddLayers[i],
                           Num_Of_variables=Num_Of_variables,
                           LR=LR,
                           architecture=architecture,
                           actFunc=actFuncs,
                           learning_rate=learning_rate[i],
                           regulizer =regulizer,
                           regulizerScale=regulizerScale,
                           num_hiddenRNN=num_hiddenRNN[i],
                           DropoutMemoryStates= DropoutMemoryStates,
                           DropoutVariational_recurrent=DropoutVariational_recurrent,
                           output_keep_prob=output_keep_prob,
                           state_keep_prob=state_keep_prob)
      
      vmodel = ValueModel(D, A, 
                           NumOfLayers=NumOfHiddLayers[i],
                           Num_Of_variables=Num_Of_variables,
                           LR=LR,
                           architecture=architecture,
                           actFunc=actFuncs,
                           learning_rate=learning_rate[i],
                           regulizer =regulizer,
                           regulizerScale=regulizerScale,
                           num_hiddenRNN=num_hiddenRNN[i],
                           DropoutMemoryStates= DropoutMemoryStates,
                           DropoutVariational_recurrent=DropoutVariational_recurrent,
                           output_keep_prob=output_keep_prob,
                           state_keep_prob=state_keep_prob)
      
      init = tf.global_variables_initializer()
      session = tf.InteractiveSession()
      session.run(init)    
      
      pmodel.set_session(session)
      vmodel.set_session(session)
      gamma = 0.95
    
      N = 10000
      sorts = np.empty(N)
      nominal_rewards = np.empty(N)
      
      t_sorts = np.empty(N)
      t_nominal_rewards = np.empty(N)
      
      for n in range(N):
        s_time = time.time()
        sort, nominal_reward, t_sort, t_nom = play_one_td([env_trading, env_testing], pmodel, vmodel, gamma)
        e_time = time.time()
        
        if isnan(sort):
            break
        
        sorts[n] = sort    
        nominal_rewards[n] = nominal_reward
        
        t_sorts[n] = t_sort    
        t_nominal_rewards[n] = t_nom
        
        if n % 1 == 0:
            print("episode:", n, 
                "total sort: %.4f" % sort,
                "nominal rewards: %.4f" % nominal_reward,
                "testing sort %.4f:" %t_sort,
                "testing nominal rewards: %.4f" % t_nom,
                "in time: %.3f" %(e_time-s_time))
        
      
      filenameModel = "/afs/inf.ed.ac.uk/user/s17/s1749290/mlp3/gym_trading/envs/saved_models/" + name[i]
        
      if not os.path.exists(filenameModel):
          os.makedirs(filenameModel)   
            
      np.savetxt(filenameModel+"/sorts.txt", sorts)
      np.savetxt(filenameModel+"/nominal_rewards.txt", nominal_rewards)
      
      np.savetxt(filenameModel+"/test_sorts.txt", t_sorts)
      np.savetxt(filenameModel+"/test_nominal_rewards.txt", t_nominal_rewards)
      
      saver = tf.train.Saver(save_relative_paths=True)
      saver.save(session, filenameModel+"/model.ckpt")
  
  #print(saved_path)
  
#  plt.plot(sorts)
#  plt.plot(nominal_rewards)
#  plt.title("Rewards")
#  plt.show()

  #plot_running_avg(totalrewards)
  #plot_cost_to_go(env, vmodel) 
    
def make_testing_predictions(env, pmodel):
    
  observation,_ = env.reset()
    
  action = pmodel.sample_action(observation)
  #action = np.array([[0,0,1]])
  #action = np.random.uniform(-1, 1, size=(1,3))
    
  observation, reward, done, sort , info, _ = env.step(action)
  #print(action)
  return sort, info['nominal_reward']

training()
#main_testing()