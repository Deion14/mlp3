
import gym
from gym import error, spaces, utils
from gym.utils import seeding
from collections import Counter

import quandl
import numpy as np
from numpy import random
import pandas as pd
import logging
import pdb
from sklearn import preprocessing
import tempfile

log = logging.getLogger(__name__)
log.info('%s logger started.',__name__)


def _sharpe(Returns, freq=252) :
  """Given a set of returns, calculates naive (rfr=0) sharpe """
  return (np.sqrt(freq) * np.mean(Returns))/np.std(Returns)

def _prices2returns(prices):
  px = pd.DataFrame(prices)
  nl = px.shift().fillna(0)
  R = ((px - nl)/nl).fillna(0).replace([np.inf, -np.inf], np.nan).dropna()
  R = np.append( R[0].values, 0)
  return R



class QuandlEnvSrcTest(object):
  ''' 
  Quandl-based implementation of a TradingEnv's data source.
  
  Pulls data from Quandl, preps for use by TradingEnv and then 
  acts as data provider for each new episode.
  '''
  q_api_key = "bB4wp5--7XrkpGZ7-gxJ"
  quandl.ApiConfig.api_key = q_api_key
  MinPercentileDays = 100 
  QuandlAuthToken = ""  # not necessary, but can be used if desired
  Name = "TSE/9994" # https://www.quandl.com/search (use 'Free' filter)

  def __init__(self, days=252, name=Name, auth=QuandlAuthToken, scale=True ):
    self.name = name
    self.auth = auth
    self.days = days+1
    log.info('getting data for %s from quandl...',QuandlEnvSrcTest.Name)
   
    
    Stocks=['A','AAPL','MSFT']
    self.NumberOfStocks=len(Stocks)
    
    df = quandl.get_table('WIKI/PRICES', ticker=Stocks, qopts = { 'columns': ['ticker', 'volume','adj_close'] }, date = { 'gte': '2016-12-25', 'lte': '2017-12-29' }, paginate=False) 
    
    df = df[ ~np.isnan(df.volume)][['ticker','volume', 'adj_close']]
    
    # we calculate returns and percentiles, then kill nans
    df = df[['ticker','adj_close','volume']] 
    self.Dimension=len(list(df))
    
    df.volume.replace(0,1,inplace=True) # days shouldn't have zero volume..
    df['Return'] = (df.adj_close-df.adj_close.shift())/df.adj_close.shift()
    #df['Return2Day'] = (df.adj_close-df.adj_close.shift(periods=2))/df.adj_close.shift(periods=2)
    #df['Return5Day'] = (df.adj_close-df.adj_close.shift(periods=5))/df.adj_close.shift(periods=5)
    #df['Return10Day'] = (df.adj_close-df.adj_close.shift(periods=10))/df.adj_close.shift(periods=10)
    pctrank = lambda x: pd.Series(x).rank(pct=True).iloc[-1]
    names=["Stock"+str(i) for i in range(1,3)]
    
    for i ,j in enumerate(Stocks):
        if i==0:
            DF=df[df['ticker'] == Stocks[i]].drop("ticker", axis=1 )
        elif i==1:
            stock1=df[df['ticker'] == Stocks[i]].drop("ticker", axis=1 )
            stock1=  stock1.set_index(np.arange(0,len(stock1)))
            DF=DF.join(stock1, lsuffix='Stock1', rsuffix='Stock2')

        else:

            stock1=df[df['ticker'] == Stocks[i]].drop("ticker", axis=1 )
            stock1=  stock1.set_index(np.arange(0,len(stock1)))
            DF=DF.join(stock1, rsuffix=names[i-2])
        
    DF=DF.iloc[1:] # remove 1st 10 
    colNames=list(DF)
    #removeRetCols = ["ReturnStock"+str(i) for i in range(1,3)]
    
    colNames = [i for j, i in enumerate(colNames) if j not in range(self.Dimension-1,self.NumberOfStocks*self.Dimension,self.Dimension)]

    DF[colNames] = DF[colNames].apply(lambda x: (x - x.mean()) / (x.var()))
    
    df=DF
    self.min_values = df.min(axis=0)
    self.max_values = df.max(axis=0)
    self.data = df
    self.step = 0


  def reset(self):
    # automatically starts at first 
    
    self.idx = 0  #np.random.randint( low = 0, high=len(self.data.index)-self.days )
    self.step = 0

  def _step(self):    

    obs = self.data.iloc[self.idx].as_matrix()
    self.idx += 1
    self.step += 1
    done = self.step >= self.days
    #pdb.set_trace()

    retAllStocks=list(np.arange(self.Dimension-1,self.Dimension*self.NumberOfStocks,self.Dimension ))
    returns=self.data.iloc[:self.idx,retAllStocks] #past returns of stocks
    
    return obs,done,returns



                    #############################                 #########################################

                    #############################                 #########################################





class TradingSim(object) :
  """ Implements core trading simulator for single-instrument univ """

  def __init__(self, steps, trading_cost_bps = 1e-3, time_cost_bps = 1e-4,NumberOfStocks=2):

    # invariant for object life
    self.NumberOfStocks   =NumberOfStocks
    self.trading_cost_bps = trading_cost_bps
    self.time_cost_bps    = time_cost_bps
    self.steps            = steps
    # change every step
    self.step             = 0
    self.actions          = np.zeros((self.steps,self.NumberOfStocks))
    self.navs             = np.ones(self.steps)
    self.mkt_nav         = np.ones(self.steps)
    self.strat_retrns     = np.ones(self.steps)
    self.posns            = np.zeros(self.steps)
    self.costs            = np.zeros(self.steps)
    self.trades           = np.zeros(self.steps)
    self.mkt_retrns       = np.zeros((self.steps,1))
    self.total_returns    = 0
    self.negative_returns = np.zeros(0)
    
  def reset(self):
    self.step = 0
    self.actions.fill(0)
    self.navs.fill(1)
    self.mkt_nav.fill(1)
    self.strat_retrns.fill(0)
    self.posns.fill(0)
    self.costs.fill(0)
    self.trades.fill(0)
    self.mkt_retrns.fill(0)
    self.total_returns    = 0
    
  def _step(self, action, retrn ):
    """ Given an action and return for prior period, calculates costs, navs,
        etc and returns the reward and a  summary of the day's activity. """

    #bod_posn = 0.0 if self.step == 0 else self.posns[self.step-1]
    #bod_nav  = 1.0 if self.step == 0 else self.navs[self.step-1]
    #mkt_nav  = 1.0 if self.step == 0 else self.mkt_nav[self.step-1]

    
    self.actions[self.step,:] = action
    #self.posns[self.step] = action - 1     
    #self.trades[self.step] = self.posns[self.step] - bod_posn

    trade_costs_pct = abs(self.trades[self.step]) * self.trading_cost_bps 
    self.costs[self.step] = trade_costs_pct +  self.time_cost_bps
    

    reward= np.dot(retrn, action.reshape(-1,1))-self.costs[self.step]

    nominal_reward = np.dot(retrn, action.reshape(-1,1)) - self.costs[self.step]
    self.total_returns = self.total_returns + nominal_reward

    oldsort = self.mkt_retrns[self.step-1,:]
    newsort = 0
    sortchange = 0
    stdev_neg_returns = 0
    self.negative_returns = [0]
    
    if nominal_reward < 0:
        self.negative_returns = np.append(self.negative_returns, nominal_reward)
        stdev_neg_returns = np.std(self.negative_returns)
    else:
        stdev_neg_returns = np.std(self.negative_returns)
        
    if stdev_neg_returns == 0:
        newsort = self.total_returns / .0001
    else:
        newsort = self.total_returns / stdev_neg_returns
    

    if oldsort == 0:
        sortchange = newsort
    else:
        sortchange = (newsort - oldsort)/oldsort

        
    self.mkt_retrns[self.step,:] = newsort
             
    
    #reward = ( (bod_posn * retrn) - self.costs[self.step] )
    #pdb.set_trace()
   # self.strat_retrns[self.step] = sortchange

    #if self.step != 0 :
    #  self.navs[self.step] =  bod_nav * (1 + self.strat_retrns[self.step-1])
    #  self.mkt_nav[self.step] =  mkt_nav * (1 + self.mkt_retrns[self.step-1])
    
    #info = { 'reward': reward, 'nav':self.navs[self.step], 'costs':self.costs[self.step] }
    info = { 'reward': reward,  'costs':self.costs[self.step] }


    self.step += 1      
    return sortchange, newsort, info


  def to_df(self):
    """returns internal state in new dataframe """
    cols = ['action', 'bod_nav', 'mkt_nav','mkt_return','sim_return',
            'position','costs', 'trade' ]
    rets = _prices2returns(self.navs)
    #pdb.set_trace()
    df = pd.DataFrame( )
    """    
        {'action':     self.actions, # today's action (from agent)
                          'bod_nav':    self.navs,    # BOD Net Asset Value (NAV)
                          'mkt_nav':  self.mkt_nav, 
                          'mkt_return': self.mkt_retrns,
                          'sim_return': self.strat_retrns,
                          'position':   self.posns,   # EOD position
                          'costs':  self.costs,   # eod costs
                          'trade':  self.trades },# eod trade
                         columns=cols)
                             """
    return df



                    #############################                 #########################################

                    #############################                 #########################################



class TestingEnv(gym.Env):
  """This gym implements a simple trading environment for reinforcement learning.

   """
  metadata = {'render.modes': ['human']}

  def __init__(self):
    self.days = 252
    self.src = QuandlEnvSrcTest(days=self.days)
    self.sim = TradingSim(steps=self.days, trading_cost_bps=1e-3,
                          time_cost_bps=1e-4,NumberOfStocks=self.src.NumberOfStocks)

    self.action_space =  spaces.Box(low=-1, high=1, shape=(self.src.NumberOfStocks,))

    self.observation_space= spaces.Box( self.src.min_values,
                                        self.src.max_values)
    self._reset()

  def _configure(self, display=None):
    self.display = display

  def _seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  def _step(self, action):
    #assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
    observation, done, Returns = self.src._step()
    retAllStocks=list(np.arange(self.src.Dimension-1,self.src.Dimension*self.src.NumberOfStocks,self.src.Dimension ))
    yret = observation[retAllStocks]
    

    reward, sort, info = self.sim._step( action, yret )

    return observation, reward, done, sort, info, Returns

  
  def _reset(self):
    self.src.reset()
    self.sim.reset()
    out=self.src._step()#,self.src._step()[2] 
    
    
    return out[0], out[2]#changes this form [0] to this 
    
  def _render(self, mode='human', close=False):
    #... TODO
    pass

  # some convenience functions:
  
  def run_strat(self,  strategy, return_df=True):
    """run provided strategy, returns dataframe with all steps"""
    observation = self.reset()
    done = False
    while not done:
      action = strategy( observation, self ) # call strategy
      observation, reward, done, info = self.step(action)

    return self.sim.to_df() if return_df else None
      
  def run_strats( self, strategy, episodes=1, write_log=True, return_df=True):
    """ run provided strategy the specified # of times, possibly
        writing a log and possibly returning a dataframe summarizing activity.
    
        Note that writing the log is expensive and returning the df is moreso.  
        For training purposes, you might not want to set both.
    """
    logfile = None
    if write_log:
      logfile = tempfile.NamedTemporaryFile(delete=False)
      log.info('writing log to %s',logfile.name)
      need_df = write_log or return_df

    alldf = None
        
    for i in range(episodes):
      df = self.run_strat(strategy, return_df=need_df)
      if write_log:
        #df.to_csv(logfile, mode='a')
        if return_df:
          alldf = df if alldf is None else pd.concat([alldf,df], axis=0)
            
    return alldf
