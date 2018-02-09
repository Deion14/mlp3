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

class PolicyGradient(object) :
    """ Policy Gradient implementation in tensor flow.
   """
    
    def __init__(self,
                 sess,                # tensorflow session
                 obs_dim,             # observation shape
                 num_actions,         # number of possible actions
                 neurons_per_dim=32,  # hidden layer will have obs_dim * neurons_per_dim neurons
                 learning_rate=1e-2,  # learning rate
                 gamma = 0.9,         # reward discounting 
                 decay = 0.9          # gradient decay rate
                 ):
                 
        self._sess = sess
        self._gamma = gamma
        self._tf_model = {}
        self._num_actions = num_actions
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
                                                   initializer=L2) '''
        ######################   
        NumberOfLayers=2
        self.NumofLayers=range(0,NumberOfLayers)
        whichLayers= ["layer_one","layer_two"]
        self.NameW=["W1", "W2"]
        InputDimensions=[obs_dim, hidden_neurons]
        OutputDimensions=[hidden_neurons, self._num_actions]
        for  i in self.NumofLayers:   
            whichLayer=whichLayers[i]
            inputsDim=InputDimensions[i]
            outputDim=OutputDimensions[i]
            NameW  =self.NameW[i]
            with tf.variable_scope(whichLayer,reuse=tf.AUTO_REUSE):
                L2 = tf.truncated_normal_initializer(mean=0,
                                                 stddev=1./np.sqrt(hidden_neurons),
                                                 dtype=tf.float32)
                self._tf_model[NameW] = tf.get_variable(NameW,
                                                   [inputsDim,outputDim],
                                                   initializer=L2)
       
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

        self._saver = tf.train.Saver()

        # tf optimizer op
        self._tf_aprob = self.tf_policy_forward(self._tf_x)
        loss = tf.nn.l2_loss(self._tf_y - self._tf_aprob) # this gradient encourages the actions taken
        optimizer = tf.train.RMSPropOptimizer(learning_rate, decay=decay)
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

    def tf_policy_forward(self, x): #x ~ [1,D]
        
        '''
        h = tf.matmul(x, self._tf_model['W1'])
        h = tf.nn.relu(h)
        logp = tf.matmul(h, self._tf_model['W2']) '''
        
        #################        #################        #################
        
        for i in self.NumofLayers:

            if i ==0:
                 print(self.NameW[i])
                 h = tf.matmul(x, self._tf_model[self.NameW[i]])
                 h = tf.nn.relu(h)
            elif i>0 and i < max(self.NumofLayers):
                 h = tf.matmul(h, self._tf_model[self.NameW[i]])
                 h = tf.nn.relu(h)
            else:
                 h = tf.matmul(h, self._tf_model[self.NameW[i]])

        
        
                #################        #################        #################
        pdb.set_trace()     
        logp=h
        sign=tf.sign(logp)
        #p=tf.sign(logp)*tf.exp(tf.abs(logp)) / tf.reduce_sum(tf.exp(tf.abs(logp)), axis=-1)
        absLogP=tf.abs(logp)

        p = tf.multiply(sign,tf.nn.softmax(absLogP))
        
        
        return p,logp
    
    
    def GaussianNoise(inputs, returns):
        
        variance=np.var(returns,axis=0)
        noise=np.random.normal(0,variance)
        t1=  inputs+noise
        output=t1/t1.sum() 
        return output
    
    
    
    def train_model(self, env, episodes=100, 
                    load_model = False,  # load model from checkpoint if available:?
                    model_dir = '/tmp/pgmodel/', log_freq=10 ) :

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
            #print(episode)
            
            aprob,logp = self._sess.run(self._tf_aprob,feed)
            #aprob = aprob[0,:] # we live in a batched world :/
            
            print(aprob)
            aprob=PolicyGradient.GaussianNoise(aprob,Returns)
            action=aprob
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
