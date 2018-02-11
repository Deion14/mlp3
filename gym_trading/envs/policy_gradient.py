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
import gym_trading

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
                 actFunc,
                 LR,
                 regulizer=None,
                 regulizerScale=0.01,
                 neurons_per_dim=32,  # hidden layer will have obs_dim * neurons_per_dim neurons
                 learning_rate=1e-2,  # learning rate
                 gamma = 0.9,         # reward discounting 
                 decay = 0.9          # gradient decay rate
                 
                 ):
        self.actFunc=actFunc
        self.Reg= regulizer
        self.RegScale= regulizerScale    
        self._sess = sess
        self.learningRule=LR
        self._gamma = gamma
        self._tf_model = {}
        self._num_actions = num_actions
        self._num_stocks = num_actions
        self.NumofLayers=NumOfLayers
        hidden_neurons = obs_dim * neurons_per_dim
        
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
            
#        pdb.set_trace()
        self.Reg= None
        self._tf_aprob = self.tf_policy_forward(self._tf_x,OutputDimensions)
        loss = tf.nn.l2_loss(self._tf_y - self._tf_aprob) # this gradient encourages the actions taken
        self._saver = tf.train.Saver()
        

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
            
        
        tf_grads = optimizer.compute_gradients(loss, var_list=tf.trainable_variables(), 
                                               grad_loss=self._tf_discounted_epr)
        tf_grads = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in tf_grads]

        self._train_op = optimizer.apply_gradients(tf_grads)
    
    def tf_discount_rewards(self, tf_r): #tf_r ~ [game_steps,1]
        discount_f = lambda a, v: a*self._gamma + v;
        tf_r_reverse = tf.scan(discount_f, tf.reverse(tf_r,[0]))
        tf_discounted_r = tf.reverse(tf_r_reverse,[0])
        tf_discounted_r = tf.clip_by_value(tf_discounted_r, -1., 1.)


        #tf_r_reverse = tf.scan(discount_f, tf.reverse(tf_r,[True, False]))
        #tf_discounted_r = tf.reverse(tf_r_reverse,[True, False])
        return tf_discounted_r

    def tf_policy_forward(self, x,OutputDimensions): #x ~ [1,D]
        
        #################        #################        #################
        '''
        h = tf.matmul(x, self._tf_model['W1'])
        h = tf.nn.relu(h)
        logp = tf.matmul(h, self._tf_model['W2']) 
        '''
        #################        #################        #################
        #################        #################        #################
        '''
        
        for i in range(0,self.NumofLayers):

            if i ==0:
                
                 h = tf.matmul(x, self._tf_model[self.NameW[i]])
                 h = tf.nn.relu(h)
            elif i>0 and i < max(range(self.NumofLayers)):
                 h = tf.matmul(h, self._tf_model[self.NameW[i]])
                 h = tf.nn.relu(h)
            else:
                 h = tf.matmul(h, self._tf_model[self.NameW[i]])
                 logp=h
        '''
        #################        #################        #################
        if self.actFunc=="softmax":
                actFunc=tf.nn.softmax
        elif  self.actFunc=="relu":
                actFunc=tf.nn.relu
        elif self.actFunc=="elu":
                actFunc= tf.nn.elu
        else:
            assert("wrong acttivation function")
                   
            
        init=tf.contrib.layers.xavier_initializer(uniform=False, seed=1, dtype=tf.float32)

        for i in range(0,self.NumofLayers):
            
            outputDim=OutputDimensions[i] #OutputDimensions[i]
           
            if i ==0:
                 h=tf.contrib.layers.fully_connected(x,
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
            elif i>0 and i < max(range(self.NumofLayers)):
                                             
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
        # last Layer to output    
                                             
        h=tf.contrib.layers.fully_connected(h,
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
        logp=h
    
        
                #################        #################        #################

        
        sign=tf.sign(logp)
        #p=tf.sign(logp)*tf.exp(tf.abs(logp)) / tf.reduce_sum(tf.exp(tf.abs(logp)), axis=-1)
        absLogP=tf.abs(logp)

        p = tf.multiply(sign,tf.nn.softmax(absLogP))
        
        
        return p,logp
    
    
    def GaussianNoise(inputs, returns):
        
        stdd=np.std(returns,axis=0)
        noise=np.random.normal(0,stdd*2)
        t1=  (inputs+noise)
        if abs(t1).sum()==0:
            print("Yup some zero variance shit here")
            output=t1
        else:
             output=t1/abs(t1).sum() 
        return output
    
    
    
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
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        episode = 0
        observation,Returns = env.reset()
        
#        x = observation[0]
#        Returns = observation[1]
        
        
        xs,rs,ys = [],[],[]    # environment info
        running_reward = 0    
        reward_sum = 0
        # training loop
        day = 0
        simrors = np.zeros(episodes)
        mktrors = np.zeros(episodes)
        alldf = None
        victory = False
        while episode < episodes and not victory:
            # stochastically sample a policy from the network

            
            x=observation
            feed = {self._tf_x: np.reshape(x, (1,-1))}
           
            
            aprob,logp = self._sess.run(self._tf_aprob,feed)
            
            #aprob = aprob[0,:] # we live in a batched world :/

            aprob=PolicyGradient.GaussianNoise(aprob,Returns)
            
            action=aprob
            #print(aprob)
            #action = np.random.choice(self._num_actions, p=aprob)
            #label = np.zeros_like(aprob) ; label[action] = 1 # make a training 'label'
            label=action
            
            # step the environment and get new measurements

            observation, reward, done, sort, info, Returns = env.step(action)

            #print observation, reward, done, info
            reward_sum += reward


            # record game history
            xs.append(x)
            ys.append(label)
            rs.append(reward)
            day += 1
            if done:
                running_reward = running_reward * 0.99 + reward_sum * 0.01
                print(action)
                epx = np.vstack(xs)
                epr = np.vstack(rs)
                epy = np.vstack(ys)

                xs,rs,ys = [],[],[] # reset game history
                #df = env.sim.to_df()
                #pdb.set_trace()
                #simrors[episode]=df.bod_nav.values[-1]-1 # compound returns
                #mktrors[episode]=df.mkt_nav.values[-1]-1

                #alldf = df if alldf is None else pd.concat([alldf,df], axis=0)
                
                feed = {self._tf_x: epx, self._tf_epr: epr, self._tf_y: epy}
                _ = self._sess.run(self._train_op,feed) # parameter update

                if episode % log_freq == 0:
                    log.info('year #%6d, mean reward: %8.4f, sim ret: %8.4f, mkt ret: %8.4f, net: %8.4f', episode,
                             sort, simrors[episode],mktrors[episode], simrors[episode]-mktrors[episode])
                    save_path = self._saver.save(self._sess, model_dir+'model.ckpt',
                                                 global_step=episode+1)
                    print(sort)
                    if episode > 100:
                        vict = pd.DataFrame( { 'sim': simrors[episode-100:episode],
                                               'mkt': mktrors[episode-100:episode] } )
                        vict['net'] = vict.sim - vict.mkt
                        if vict.net.mean() > 0.0:
                            victory = True
                            log.info('Congratulations, Warren Buffet!  You won the trading game.')
                    #print("Model saved in file: {}".format(save_path))

                
                    
                episode += 1
                observation,Returns = env.reset()
                reward_sum = 0
                day = 0
                
        return alldf, pd.DataFrame({'simror':simrors,'mktror':mktrors})
