''' Policy Gradient implementation customized a bit for 
solving the trading problem'''
# stolen shamelessly and adapted December 2016 by Tito Ingargiola
# was originally:

'''Solution to the Cartpole problem using Policy Gradients in Tensorflow.'''
# written October 2016 by Sam Greydanus
# inspired by gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5

import numpy as np
import gym
import tensorflow as tf
import pdb
import logging
import os.path
import pandas as pd
import time
import gym_trading
import pickle as pkl
import os
import trading_env as te

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
log.info('%s logger started.',__name__)
np.seterr(all='raise')
class PolicyGradient(object) :
    """ Policy Gradient implementation in tensor flow.
   """
    
    def __init__(self,
                 sess,                # tensorflow session
                 obs_dim,             # observation shape
                 num_actions,         # number of possible actions
                 NumOfLayers,
                 Num_Of_variables,
                 LR,
                 architecture,
                 actFunc,
                 avgfilename,          #name for pickle file of average values
                 Modelfilename,        #name for model to save   
                 num_hiddenRNN=24,
                 regulizer=None,
                 regulizerScale=0.01,
                 neurons_per_dim=32,  # hidden layer will have obs_dim * neurons_per_dim neurons
                 learning_rate=1e-2,  # learning rate
                 gamma = 0.9,         # reward discounting 
                 decay = 0.9,          # gradient decay rate
                 DropoutMemoryStates=None,
                 DropoutVariational_recurrent=False,
                 state_keep_prob=1,
                 output_keep_prob=1
                 
                 ):
        self.actFunc=actFunc
        self.obs_dim=obs_dim
        self.Reg= regulizer
        self.RegScale= regulizerScale    
        self._sess = sess
        self.learningRule=LR
        self._gamma = gamma
        self.num_hiddenRNN=num_hiddenRNN
        self._tf_model = {}
        self._num_actions = num_actions
        self._num_stocks = num_actions
        self._variables=Num_Of_variables
        self.NumofLayers=NumOfLayers   # NUM OF HIDDEN LAYERS NOW!!!!!
        hidden_neurons = obs_dim * neurons_per_dim
        self.architecture = architecture
        self.last100avg = []
        self.filename = avgfilename
        self.filenameModel= Modelfilename
        self.DropoutMemoryStates=DropoutMemoryStates
        self.DropoutVariational_recurrent=DropoutVariational_recurrent,
        self.output_keep_prob=output_keep_prob
        self.state_keep_prob=state_keep_prob
#        tf.set_random_seed(1234)
        
        '''
        
        with tf.variable_scope('layer_one',reuse=tf.AUTO_REUSE):
            L1 = tf.truncated_normal_initializer(mean=0,
                                                 stddev=1./np.sqrt(obs_dim),
                                                 dtype=tf.float32)
            self._tf_model['W1'] = tf.get_variable("W1",
                                                   [obs_dim, hidden_neurons],
                                                   initializer=L1)
        with tf.variable_scope('layer_two',reuse=tf.AUTO_REUSE):
            L2 = tf.truncated_normal_initializer(mean=0,
                                                 stddev=1./np.sqrt(hidden_neurons),
                                                 dtype=tf.float32)
            self._tf_model['W2'] = tf.get_variable("W2",
                                                   [hidden_neurons,num_actions],
                                                   initializer=L2) 
        ######################   
        
        '''
        
        '''
        whichLayers=["layer_"+str(i) for i in range(1, self.NumofLayers+1)]
        self.NameW=["W"+str(i) for i in range(1, self.NumofLayers+1)]
        InputDimensions=[obs_dim]+ [hidden_neurons for i in range(0, self.NumofLayers-1)]
        OutputDimensions=[hidden_neurons for i in range(0, self.NumofLayers-1)]+[self._num_actions]
        for  i in range(self.NumofLayers):   
            whichLayer=whichLayers[i]
            inputsDim=InputDimensions[i]
            outputDim=OutputDimensions[i]
            NameW  =self.NameW[i]
            with tf.variable_scope(whichLayer,reuse=tf.AUTO_REUSE):
                L2 = tf.contrib.layers.xavier_initializer(uniform=False, seed=1, dtype=tf.float32)
                #tf.truncated_normal_initializer(mean=0,
                      #                           stddev=1./np.sqrt(hidden_neurons),
                       #                          dtype=tf.float32)
                self._tf_model[NameW] = tf.get_variable(NameW,
                                                   [inputsDim,outputDim],
                                                   initializer=L2)
        '''
        
        # tf placeholders
        self._tf_x = tf.placeholder(dtype=tf.float32, shape=[None, obs_dim],name="tf_x")
        self._tf_y = tf.placeholder(dtype=tf.float32, shape=[None, num_actions],name="tf_y")
        self._tf_epr = tf.placeholder(dtype=tf.float32, shape=[None,1], name="tf_epr")
        self.X = tf.placeholder(tf.float32, shape=(None, 252, 30), name='X_for_policy')
        self.actions = tf.placeholder(tf.float32, shape=(None,2), name='actions')
        self.conv = tf.placeholder(tf.float32, shape=(None), name='conv')
        
        # tf reward processing (need tf_discounted_epr for policy gradient wizardry)
        self._tf_discounted_epr = self.tf_discount_rewards(self._tf_epr)
        self._tf_mean, self._tf_variance= tf.nn.moments(self._tf_discounted_epr, [0], 
                                                        shift=None, name="reward_moments")
        self._tf_discounted_epr -= self._tf_mean
        self._tf_discounted_epr /= tf.sqrt(self._tf_variance + 1e-6)

        #self._saver = tf.train.Saver()

        # tf optimizer op
        OutputDimensions=[hidden_neurons for i in range(0, self.NumofLayers-1)]+[self._num_actions]
        
        # Different Regularization Rules            
        if self.Reg=="l2":
                self.Reg = tf.contrib.layers.l2_regularizer(scale=self.RegScale)
        elif  self.Reg=="l1":
                self.Reg= tf.contrib.layers.l1_regularizer(scale=self.RegScale)
        elif  self.Reg=="None":
                self.Reg= None
        else:
            assert("wrong acttivation function")       
            
        
        self._tf_aprob = self.tf_policy_forward(self.X,OutputDimensions)
       
        loss = tf.nn.l2_loss(self._tf_y - self._tf_aprob) # this gradient encourages the actions taken
       
        self._saver = tf.train.Saver(save_relative_paths=True)
        
        # Different Learning Rules
        if self.learningRule=="RMSProp":
            optimizer = tf.train.RMSPropOptimizer(learning_rate, decay=decay)
        elif self.learningRule=="Adam":
            optimizer = tf.train.AdamOptimizer(learning_rate,
                                               beta1=0.9,
                                               beta2=0.999)
        elif self.learningRule=="Mom":
            optimizer = tf.train.MomentumOptimizer(learning_rate,
                                                momentum=0.8,
                                                use_locking=False,
                                                use_nesterov=True)
        elif self.learningRule=="GD":
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            
        #self._tf_discounted_epr = self.get_grad_loss(self._tf_epr, loss)
        self._tf_discounted_epr = self.get_grad_loss(self._tf_discounted_epr, loss)
        
        
        tf_grads = optimizer.compute_gradients(loss, var_list=tf.trainable_variables(), 
                                               grad_loss=self._tf_discounted_epr)
        def ClipIfNotNone(grad):
            if grad is None:
                return grad
            return tf.clip_by_value(grad, -1.2, 1.2)
        tf_grads_clipped = [(ClipIfNotNone(grad), var) for grad, var in tf_grads]

        self.tf_grads = tf_grads
        self.tf_grads_clipped = tf_grads_clipped
        self._train_op = optimizer.apply_gradients(tf_grads)
    
    def get_grads_and_clipping(self):
        return self.tf_grads, self.tf_grads_clipped
    
    def get_grad_loss(self, tf_r, diff):
        grad = tf.multiply(tf_r, diff)
        discount_f = lambda a, v: a*(1/v)*(1/2)
        grad_loss = tf.scan(discount_f, grad, self.conv)
        
        return grad_loss
    
    def tf_discount_rewards(self, tf_r): #tf_r ~ [game_steps,1]
        discount_f = lambda a, v: a*self._gamma + v;
        tf_r_reverse = tf.scan(discount_f, tf.reverse(tf_r,[0]))
        tf_discounted_r = tf.reverse(tf_r_reverse,[0])
        #tf_discounted_r = tf.clip_by_value(tf_discounted_r, -1.2, 1.2)


        #tf_r_reverse = tf.scan(discount_f, tf.reverse(tf_r,[True, False]))
        #tf_discounted_r = tf.reverse(tf_r_reverse,[True, False])
        return tf_discounted_r

    def tf_policy_forward(self, x,OutputDimensions): #x ~ [1,D]
        
        #################        #################        #################

        if self.actFunc=="sigmoid":
                actFunc=tf.nn.sigmoid
        elif  self.actFunc=="relu":
                actFunc=tf.nn.relu
        elif self.actFunc=="lrelu":
                actFunc= tf.nn.leaky_relu
        elif self.actFunc=="elu":
                actFunc= tf.nn.elu   
        elif self.actFunc=="selu":
                actFunc= tf.nn.selu                   
        else:
            assert("wrong acttivation function")
                    
        init=tf.contrib.layers.xavier_initializer(uniform=False, dtype=tf.float32)
                  
        def dropout_state_filter_visitors(state): 
            ''' Dropout of memory cells Not literature based on tensorflow code'''
            
            if isinstance(state, tf.contrib.rnn.LSTMStateTuple): # Never perform dropout on the c state.
                return tf.contrib.rnn.LSTMStateTuple(c=True, h=True) 
            elif isinstance(state, tf.TensorArray): 
                return False 
            return True

        
        
        if self.architecture != "FFNN":
            def get_a_cell(num_hidden,i):
                    ''' Function for GRU, RNN, LSTM'''

                    cell_type=self.architecture

                    if cell_type == 'GRU':
                        cell = tf.nn.rnn_cell.GRUCell(num_hidden,activation=actFunc)
                    elif cell_type == 'LSTM':
                        cell = tf.nn.rnn_cell.LSTMCell(num_hidden,activation=actFunc, state_is_tuple=True)
                    elif cell_type == 'RNN':
                        cell = tf.nn.rnn_cell.BasicRNNCell(num_hidden,activation=actFunc)

#                    pdb.set_trace()

                    drop = tf.nn.rnn_cell.DropoutWrapper(cell, 
                                                         input_keep_prob=1,
                                                         output_keep_prob=self.output_keep_prob,
                                                         state_keep_prob=self.state_keep_prob,
                                                         variational_recurrent=self.DropoutVariational_recurrent,
                                                         input_size=self._num_stocks*self._variables if i==0 else tf.TensorShape(num_hidden), 
                                                         dtype=tf.float32,
                                                         seed=None,
                                                         dropout_state_filter_visitor=dropout_state_filter_visitors if self.DropoutMemoryStates==True else None                                                               )
    
                    return drop

                
            ''' Create Stacked Model '''    
            with tf.name_scope('lstm'):
                  cell = tf.nn.rnn_cell.MultiRNNCell(
                  [get_a_cell(self.num_hiddenRNN, i) for i in range(self.NumofLayers)])
            ''' Make it runable '''
            with tf.variable_scope('RNN', initializer=tf.contrib.layers.xavier_initializer()):
                h, _ =  tf.nn.dynamic_rnn(cell, x, dtype=tf.float32) 
                
                  
        else:  
             ''' FFN  Has not been tested '''
            
                      ####################      ####################      ###################
                
            
           
             for i in range(0,self.NumofLayers):
            
                outputDim=OutputDimensions[i] #OutputDimensions[i]
           
                #if i ==0 and self.architecture == "LSTM":
                #
                #     h, _  = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)

                if i ==0 and self.architecture == "FFNN":
                     h=tf.contrib.layers.fully_connected(h,
                                                        outputDim,
                                                        activation_fn=actFunc,
                                                        normalizer_fn=None,
                                                        normalizer_params=None,
                                                        weights_initializer=init,
                                                        weights_regularizer=self.Reg,
                                                        biases_initializer=tf.zeros_initializer(),
                                                        biases_regularizer=None,
                                                        reuse=None,
                                                        variables_collections=None,
                                                        outputs_collections=None,
                                                        trainable=True,
                                                        scope=None)


                elif i>0 and i < max(range(self.NumofLayers)) and self.architecture == "FFNN":
                     h=tf.contrib.layers.fully_connected(h,
                                                        outputDim,
                                                        activation_fn=actFunc,
                                                        normalizer_fn=None,
                                                        normalizer_params=None,
                                                        weights_initializer=init,
                                                        weights_regularizer=self.Reg,
                                                        biases_initializer=tf.zeros_initializer(),
                                                        biases_regularizer=None,
                                                        reuse=None,
                                                        variables_collections=None,
                                                        outputs_collections=None,
                                                        trainable=True,
                                                        scope=None)       

        
          ####################      ####################      ####################
        ''' last Layer to output aka softmax shit    '''
        outputDim=10

        aaa=self.num_hiddenRNN*252
        h0 = tf.reshape(h, [-1,aaa ])
        h1=tf.contrib.layers.fully_connected(h0,#tf.contrib.layers.flatten(h),
                                                    outputDim,
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
                                                    trainable=True,
                                                    scope=None)       
        logp=h1
    
                #################        #################        #################

        
        sign=tf.sign(logp)
        absLogP=tf.abs(logp)

        p = tf.multiply(sign,tf.nn.softmax(absLogP))
        
        return p
    
    
    def GaussianNoise(inputs, returns):
        
        stdd=np.std(returns,axis=0)
        noise=np.random.normal(0,stdd*2)
        
        t1=  (inputs+noise)
        if abs(t1).sum()==0:
            output=t1
        else:
             output=t1/abs(t1).sum() 
        return output, stdd
    
    
    
    def train_model(self, env, episodes=100, 
                    load_model = False,  # load model from checkpoint if available:?
                    model_dir = "/Users/andrewplaate/mlp3/SavedModels/", log_freq=10 ) :
                   

        # initialize variables and load model
        init_op = tf.global_variables_initializer()
        self._sess.run(init_op)
        
        if load_model:
            ckpt = tf.train.get_checkpoint_state(model_dir)
            print(tf.train.latest_checkpoint(model_dir))
            if ckpt and ckpt.model_checkpoint_path:
                savr = tf.train.import_meta_graph(ckpt.model_checkpoint_path+'.meta')
                out = savr.restore(self._sess, ckpt.model_checkpoint_path)
                print("Model restored from ",ckpt.model_checkpoint_path)
            else:
                print('No checkpoint found at: ',model_dir)
  
        episode = 0
        observation,Returns = env.reset()
        
        
        xs,rs,ys = [],[],[]    # environment info
        running_reward = 0    
        reward_sum = 0
        # training loop
        day = 0
        simrors = np.zeros(episodes)
        mktrors = np.zeros(episodes)
        alldf = None
        victory = False
        
        self.sort = np.array([])
        self.NomReward = np.array([])
        
        self.mean_sort = np.array([])
        t=time.time()           
        while episode < episodes and not victory:
            # stochastically sample a policy from the network

            
            x=observation
            WIDTH= self._variables*self._num_stocks
            feed = {self.X: np.reshape(x, (-1, 252, WIDTH))}
           
            aprob = self._sess.run(self._tf_aprob,feed)
            #pdb.set_trace()
           
            aprob, std=PolicyGradient.GaussianNoise(aprob,Returns)
            action=aprob
            #action = np.random.choice(self._num_actions, p=aprob)
            #label = np.zeros_like(aprob) ; label[action] = 1 # make a training 'label'
            label=action
     
           # step the environment and get new measurements
            observation, reward, done, sort, info, Returns = env.step(action)
            nominal_reward=info["nominal_reward"]

            reward_sum += reward


            # record game history
            xs.append(x)
            ys.append(label)
            rs.append(reward)
            day += 1
            done = True
            if done:
                print(time.time()-t)    
                t=time.time()
                running_reward = running_reward * 0.99 + reward_sum * 0.01
                #epx = np.vstack(xs)
                #epx = observation
                #epX = np.reshape(np.vstack(xs), (-1, 252, WIDTH))
                epX = x

                #epr = np.vstack(rs)
                #epy = np.vstack(ys)
                epr = reward.reshape(252,1)
                epy = label

                self.NomReward = np.append(self.NomReward, nominal_reward)
                self.sort = np.append(self.sort, sort)
                #pdb.set_trace()
                xs,rs,ys = [],[],[] # reset game history
  
        
                #alldf = df if alldf is None else pd.concat([alldf,df], axis=0)
                
                feed = {self.X: epX, self._tf_epr: epr, self._tf_y: epy, self.conv: std}
                _ = self._sess.run(self._train_op,feed) # parameter update
                if episode % log_freq == 0:
                    log.info('year #%6d, mean reward: %8.4f, sim ret: %8.4f, mkt ret: %8.4f, net: %8.4f', episode,
                             sort, simrors[episode],mktrors[episode], simrors[episode]-mktrors[episode])
                if episode == episodes-1:
                    if not os.path.exists(self.filenameModel):
                       os.makedirs(self.filenameModel)
                    save_path = self._saver.save(self._sess, self.filenameModel+'/model.ckpt',
                                                 global_step=episode+1)
                
                    
                episode += 1
                observation,Returns = env.reset()
                reward_sum = 0
                day = 0
        #pdb.set_trace()        
        Sort_Returns=  np.vstack([self.sort, self.NomReward])
        
        #pkl.dump(Sort_Returns, open( self.filename, 'wb'))   
        
        return alldf, pd.DataFrame({'simror':simrors,'mktror':mktrors})
    
    
    
    
    
    
    
    
    
    
####################################  TEST          ####################################   
                    ####################################


    def test_model(self, env, episodes=100, 
                    load_model = True,  # load model from checkpoint if available:?
                    model_dir = "", log_freq=10 ) :
                   

        # initialize variables and load model
        init_op = tf.global_variables_initializer()
        self._sess.run(init_op)
        
        if load_model:
            ckpt = tf.train.get_checkpoint_state(model_dir)
            print(tf.train.latest_checkpoint(model_dir))
            if ckpt and ckpt.model_checkpoint_path:
                savr = tf.train.import_meta_graph(ckpt.model_checkpoint_path+'.meta')
                out = savr.restore(self._sess, ckpt.model_checkpoint_path)
                print("Model restored from ",ckpt.model_checkpoint_path)
            else:
                print('No checkpoint found at: ',model_dir)
        else:
                print('No model Loaded :-(')
        
        
        
        '''                            TEST CODE               '''
        
        
        
        
        episode = 0
        observation,Returns = env.reset()
        
        xs,rs,ys = [],[],[]    # environment info
        running_reward = 0    
        reward_sum = 0
        # training loop
        day = 0
        simrors = np.zeros(episodes)
        mktrors = np.zeros(episodes)
        alldf = None
        victory = False
        
        self.sort = np.array([])
        self.NomReward = np.array([])
        
        self.mean_sort = np.array([])
        t=time.time()           
        while episode < episodes and not victory:
            # stochastically sample a policy from the network
            
            
            '''                                 TEST CODE               '''
            
            
            
            x=observation
            #feed = {self._tf_x: np.reshape(x, (1,-1)),self.X: np.reshape(x, (10, 252, 3))}

            feed = {self.X: np.reshape(x, (-1, 252, 30))}
          
            aprob,logp = self._sess.run(self._tf_aprob,feed)
            action=aprob
            label=action

            observation, reward, done, sort, info, Returns = env.step(action)
            nominal_reward=info["nominal_reward"]
            #print(nominal_reward)
            #print observation, reward, done, info
            reward_sum += reward


            # record game history
            #xs.append(x)
            ys.append(label)
            rs.append(reward)
            day += 1
            if done:
                print(time.time()-t)    
                t=time.time()
                running_reward = running_reward * 0.99 + reward_sum * 0.01
                #epx = np.vstack(xs)
                epx = observation
                #epX = np.reshape(np.vstack(xs), (10, -1, 3))
                epX = np.reshape(observation, (-1,252,30))
                epr = np.vstack(rs)
                epy = np.vstack(ys)
                
                
                '''                                        TEST CODE               '''
                
                
                
                self.NomReward = np.append(self.NomReward, nominal_reward)
                self.sort = np.append(self.sort, sort)
               
                xs,rs,ys = [],[],[] # reset game history
                feed = {self.X: epX, self._tf_epr: epr, self._tf_y: epy, self._tf_x: epx}
                _ = self._sess.run(self._train_op,feed) # parameter update

                if episode % log_freq == 0:
                    log.info('year #%6d, mean reward: %8.4f, sim ret: %8.4f, mkt ret: %8.4f, net: %8.4f', episode,
                             sort, simrors[episode],mktrors[episode], simrors[episode]-mktrors[episode])
                               
                episode += 1
                observation,Returns = env.reset()
                reward_sum = 0
                day = 0
        #pdb.set_trace()        
        Sort_Returns=  np.vstack([self.sort, self.NomReward])
        
        pkl.dump(Sort_Returns, open( self.filename, 'wb'))   
        
        return alldf, pd.DataFrame({'simror':simrors,'mktror':mktrors})
   