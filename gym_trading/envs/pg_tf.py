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
  def __init__(self, D, A):
      
      
    self.D = D
    self.A = A
    self.T = 252    
    
    #initilize RNN
    num_hidden = 12
    policy_cell = tf.nn.rnn_cell.BasicRNNCell(num_hidden)
    
    # inputs and targets
    self.X = tf.placeholder(tf.float32, shape=(None, self.T, self.D), name='X_for_policy')
    self.actions = tf.placeholder(tf.float32, shape=(None,self.A), name='actions')
    self.advantages = tf.placeholder(tf.float32, shape=(None,self.A), name='advantages')
    
    # get final hidden layer
    with tf.variable_scope('policy_rnn', reuse=tf.AUTO_REUSE): 
        p_val, _  = tf.nn.dynamic_rnn(policy_cell, self.X, dtype=tf.float32)
        
    #p_val = tf.transpose(p_val, [1, 0, 2])
    #last = tf.gather(p_val, int(p_val.get_shape()[0]) - 1)
    
    p_val = tf.reshape(p_val,[-1,num_hidden*self.T])
#    
#    mean = mean_layer.forward(last)
#    stdv = stdv_layer.forward(last) + 1e-5 # smoothing
#    
#    # make them 1-D
#    mean = tf.reshape(mean, [-1])
#    stdv = tf.reshape(stdv, [-1])     
#    
#    norm = tf.contrib.distributions.Normal(mean, stdv)
#    
#    self.predict_op = tf.clip_by_value(norm.sample(), -1, 1)
#    
#    log_probs = norm.log_prob(self.actions)
    init = tf.contrib.layers.xavier_initializer(uniform=False, dtype=tf.float32)
    output=tf.contrib.layers.fully_connected(p_val,#tf.contrib.layers.flatten(h),
                                         self.A,
                                         activation_fn=None,
                                         normalizer_fn=None,
                                         normalizer_params=None,
                                         weights_initializer=init,
                                         weights_regularizer=self.Reg,
                                         biases_initializer=tf.zeros_initializer(),
                                         biases_regularizer=None,
                                         reuse=None,
                                         variables_collections=None,
                                         outputs_collections=None,
                                         trainable=True,scope=None)
    
    sign=tf.sign(output)
    absLogP=tf.abs(output)
    P = tf.multiply(sign,tf.nn.softmax(absLogP))
    
    # calculate output and cost
    mean = self.mean_layer.forward(P)
    stdv = self.stdv_layer.forward(P) + 1e-5 # smoothing

    # make them 1-D
    mean = tf.reshape(mean, [-1])
    stdv = tf.reshape(stdv, [-1]) 

    norm = tf.contrib.distributions.Normal(mean, stdv)
    
    log_probs = norm.log_prob(self.actions)
    
    cost = -tf.reduce_sum(self.advantages * log_probs + 0.1*norm.entropy())
    self.train_op = tf.train.AdamOptimizer(1e-3).minimize(cost)
    

  def set_session(self, session):
    self.session = session

  def partial_fit(self, X, actions, advantages):
    
    X = np.reshape(X, (-1, self.T, self.D))
    
    actions = np.reshape(actions, (-1, self.A))
    advantages = np.reshape(advantages, (-1, self.A))
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
    X = np.reshape(X, (-1, self.T, self.D))  
    p = self.predict(X)
    return p


# approximates V(s)
class ValueModel:
  def __init__(self, D, A):
    
    self.D = D
    self.A = A
    self.T = 252
    self.costs = []

    #initilize RNN
    num_hidden = 12
    value_cell = tf.nn.rnn_cell.BasicRNNCell(num_hidden)

    # inputs and targets
    self.X = tf.placeholder(tf.float32, shape=(None, self.T, self.D), name='X_for_value')
    self.Y = tf.placeholder(tf.float32, shape=(None,self.A), name='Y')
    
    value_weight = tf.Variable(tf.truncated_normal([num_hidden, self.A]))
    value_bias = tf.Variable(tf.constant(0.1, shape=[self.A]))
    
    # get final hidden layer
    with tf.variable_scope('value_rnn', reuse=tf.AUTO_REUSE): 
        v_val, _  = tf.nn.dynamic_rnn(value_cell, self.X, dtype=tf.float32)
    v_val = tf.transpose(v_val, [1, 0, 2])
    last = tf.gather(v_val, int(v_val.get_shape()[0]) - 1)
    
    Y_hat = tf.matmul(last, value_weight) + value_bias
    self.predict_op = Y_hat

    cost = tf.reduce_sum(tf.square(self.Y - Y_hat))
    self.cost = cost
    self.train_op = tf.train.AdamOptimizer(1e-1).minimize(cost)

  def set_session(self, session):
    self.session = session

  def partial_fit(self, X, Y):

    X = np.reshape(X, (-1, self.T, self.D))
    Y = np.reshape(Y, (-1, self.A))
    self.session.run(self.train_op, feed_dict={self.X: X, self.Y: Y})
    cost = self.session.run(self.cost, feed_dict={self.X: X, self.Y: Y})
    self.costs.append(cost)

  def predict(self, X):

    X = np.reshape(X, (-1, self.T, self.D))
    return self.session.run(self.predict_op, feed_dict={self.X: X})


def play_one_td(env, pmodel, vmodel, gamma):

  observation,_ = env.reset()
  done = False
  
  observations = np.zeros((252,252,30))
  actions = np.zeros((252,10))
  advantages = np.zeros((252,10))
  Gs = np.zeros((252,10))
  
  i = 0
  
  while not done:
    
    action = pmodel.sample_action(observation)
    
    prev_observation = observation
    observation, reward, done, sort , info, _ = env.step(action)
  
    # update the models
    V_next = vmodel.predict(observation)
    G = reward + gamma*V_next
    advantage = G - vmodel.predict(prev_observation)
    
    
    observations[i,:,:] = prev_observation
    actions[i,:] = action
    advantages[i,:] = advantage
    Gs[i,:] = G     

    i = i+1;
  
  pmodel.partial_fit(observations, actions, advantages)
  vmodel.partial_fit(observations, Gs)
    
  #return np.array(total_rewards[-150:]).mean(), iters
  return sort, info['nominal_reward']
  

def main_training():
    
  env = gym.make('trading-v0')
  env = env.unwrapped
  
  #D = ft.dimensions
  D,A = 30, 10
  pmodel = PolicyModel(D, A)
  vmodel = ValueModel(D, A)
  
  init = tf.global_variables_initializer()
  session = tf.InteractiveSession()
  session.run(init)    
  
  pmodel.set_session(session)
  vmodel.set_session(session)
  gamma = 0.95

  if 'monitor' in sys.argv:
    filename = os.path.basename(__file__).split('.')[0]
    monitor_dir = './' + filename + '_' + str(datetime.now())
    env = wrappers.Monitor(env, monitor_dir)

  N = 5000
  sorts = np.empty(N)
  nominal_rewards = np.empty(N)
  for n in range(N):
    sort, nominal_reward = play_one_td(env, pmodel, vmodel, gamma)
    
    sorts[n] = sort    
    nominal_rewards[n] = nominal_reward
    
    if n % 1 == 0:
      print("episode:", n, "total reward: %.4f" % sort, "avg reward (last 10): %.4f" % sorts[max(0, n-10):(n+1)].mean())

  np.savetxt("saved_models/adam_72/sorts.txt", sorts)
  np.savetxt("saved_models/adam_72/nominal_rewards.txt", nominal_rewards)
  
  saver = tf.train.Saver()
  saver.save(session, "saved_models/adam_72/model.ckpt")
  
  #print(saved_path)
  
  plt.plot(sorts)
  plt.plot(nominal_rewards)
  plt.title("Rewards")
  plt.show()

  #plot_running_avg(totalrewards)
  #plot_cost_to_go(env, vmodel) 
    
def make_testing_predictions(env, pmodel):
    
  observation,_ = env.reset()
    
  action = pmodel.sample_action(observation)
  #action = np.array([[0,0,1]])
  #action = np.random.uniform(-1, 1, size=(1,3))
    
  observation, reward, done, sort , info, _ = env.step(action)
  #print(action)
  totalreward = sort
  return totalreward    
  
def main_testing():
    
  env_testing = gym.make('testing-v0')
  env_testing = env_testing.unwrapped
  
  tf.reset_default_graph()
  
  D, A = 30,10
  pmodel = PolicyModel(D, A)
  vmodel = ValueModel(D, A)  
  
  saver = tf.train.Saver()
  session = tf.InteractiveSession()
  
  saver.restore(session, "saved_models/model.ckpt")
  
  pmodel.set_session(session)
  vmodel.set_session(session)
  
  totalrewards = []
  for i in range(50):
      totalreward = make_testing_predictions(env_testing, pmodel)
      totalrewards.append(totalreward)
  
  print(totalrewards)    

if __name__ == '__main__':
  
  s_time = time.time()
  main_training()
  e_time = time.time()
  
  print(e_time-s_time)
  #main_testing()