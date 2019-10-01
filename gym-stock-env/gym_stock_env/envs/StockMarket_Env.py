from collections import deque
import collections

import gym
from gym import spaces
import numpy as np
import pandas as pd
import sklearn
from sklearn import preprocessing

from gym_stock_env.envs.BrokerTrading_Comm import StockMarket_Comm

INITIAL_BALANCE = 10000
BUYING_POWER_MULTIPLIER = 4  # how many times is the buying power of the account, ex: x4 account balance
SEQ_LEN = 60  # minutes; is how many minutes on the past are visible on the environment

NUM_OF_SHARES_TO_BUY = 1
USE_ALL_BALANCE = False  # should we use all the account balance to buy stock?
USE_MAX_BUYING_POWER = False  # using the x 4 balance that broker gives you

MAX_STEPS = 20000


class StockMarketEnv_Train(gym.Env):
    """A stock trading environment for OpenAI gym
        this is just the training environment. Since we can 'know' the future, it is an advantage to teach the agents
        to buy and sell in the right moment.

        Should communicate with a py file that does the communication with the broker api, or a simulation of one
    """
    metadata = {'render.modes': ['human']}

    def __init__(self):
        # df = data frame from pandas -- stock data
        """
        :param df: raw input data frame.
        Should contain
            - minute of the day ( 0 - 60*24)
            - Open
            - High
            - Low
            - Close
        """

        super(StockMarketEnv_Train, self).__init__()

        self.balance = INITIAL_BALANCE
        self.shares_held = 0
        self.current_step = 0

        self.able_to_buy = True
        self.able_to_sell = False
        self.last_bought_price = 0

        """Initialize actions_space, observation_space, reward_range"""

        self.action_space = spaces.Discrete(3)  # Buy, Hold, Sell
        self.observation_space = spaces.Box(low=0, high=1, shape=(7,), dtype=np.float32)

        self.trader = None
        self.df = None

        self.raw_sequential_data = None
        self.heading_data = None

        self.sequential_data = None
        self.current_price = None
        self.current_state = None

    def init(self, df):
        self.trader = StockMarket_Comm(df)

        self.df = df
        self.raw_sequential_data = deque(maxlen=SEQ_LEN)
        self.heading_data = ["Unix Timestamp", "Open", "High", "Low", "Close", "Volume"]

        temp_list = self.trader.getPastData(SEQ_LEN)

        for i in temp_list:
            self.raw_sequential_data.append(i)
        # setup raw_sequential_data with trader data
        # self.raw_sequential_data.insert(0, self.trader.getPastData(SEQ_LEN))

        self.sequential_data = pd.DataFrame  # this is the data that is able to been seen on the environment

        self.preprocess_df()  # preprocess data and store it in sequential_data as a Pandas data frame

        self.current_price = self.trader.getCurrentPrice()

        self.current_step = SEQ_LEN  # represent the current time (index)
        self.current_state = 0  # this simulate real time, therefore the only data available is the 'Open' price

    def preprocess_df(self):
        temp_df = pd.DataFrame(self.raw_sequential_data, columns=self.heading_data)
        temp_df.dropna(inplace=True)

        col_to_change = ["Open", "High", "Low", "Close", "Volume"]

        for col in temp_df.columns:
            if col != "Unix Timestamp":
                x = temp_df[[col]].values.astype(float)
                min_max_scaler = preprocessing.MinMaxScaler()

                x_scaled = min_max_scaler.fit_transform(x)

                temp_df.drop(columns=col, inplace=True)
                temp_df[col] = x_scaled

        temp_df.dropna(inplace=True)  # fix data if there are N/A values in it

        self.sequential_data = temp_df

    def get_obs(self):
        """return the observation that the agent is able to see:

            - like stock data for the last 24 hours
                - Open, High, Low, Close
                    - it might just be one or all of them

            - the volume for the last 24 hours
            - the current time ( on hopes that it find patterns on the time of day)
            - --future-- data from other stock linked with this one

            data should get normalize so is a % of change, this will allow the agent to be able to work on different
            stocks or currencies, rather than just the one it trained on.
        """

        obs = np.array([
            self.sequential_data["Unix Timestamp"].values,
            self.sequential_data["Open"].values,
            self.sequential_data["High"].values,
            self.sequential_data["Low"].values,
            self.sequential_data["Close"].values,
            self.sequential_data["Volume"].values,
            self.trader.getCurrentPrice()
        ])

        return obs

    def step(self, action):
        """here is where the action taken by the agent is executed and return a reward and state from the action
        taken

        actions
        num     name
        0       Buy
        1       Hold
        2       Sell

        since is design to buy and sell its max possible stocks, it can't buy after buying without selling before hand
        and vice versa, it can't sell after selling, also there will be no stocks left in the later.

        Rewards

        whenever
            - Buy = 20 - (% difference to the local min * 100)
            - Sell = 20 - (% difference to the local max * 100)
                - Profit = (% of sell - buy) if > 1.8%  => + 100

        When the agent sent a buy action the env will analyze how far from the optimal buying position is the agent.
        the 20 is to make is positive even if is not buying on the optimal position, so encourage it to buy and optimize
        for a better reward (the closest to the ideal buying position the higher the reward). Same happens to the sell
        but instead of optimizing for a local minimum it optimize for a local maximum. then after a Sell action a profit
        reward is also return, if the profit is higher than 1.8% ( or another value) it will add 100 extra point, the
        intention is to encourage the agent to hold the stock for longer to get a much better reward, and maybe to learn
        to just enter positions that have higher return on investment, not just a penny per trade.


        """
        reward = 0

        if action == 0:
            # Buy
            if self.able_to_buy:
                self.trader.buyMax(True)

                self.able_to_buy = False
                self.able_to_sell = True
                self.last_bought_price = self.trader.getCurrentPrice()

                """Need to calculate reward"""

                reward += 10 - self.trader.distanceFromLocalMin()
            else:
                """ calculate the negative reward for wasting time"""
                reward -= 5
        elif action == 1:
            # Sell
            if self.able_to_sell:
                self.trader.sellALl()

                self.able_to_buy = True
                self.able_to_sell = False
                """Need to calculate reward"""

                reward += 10 - self.trader.distanceFromLocalMax()

                # profit reward
                sell_price = self.trader.getCurrentPrice()

                gain = (sell_price - self.last_bought_price) / self.last_bought_price
                reward += gain * 1500

                if gain > 0.018:
                    reward += 100

            else:
                """calculate the negative reward for wasting time"""
                reward -= 5

        done = self.trader.broke()

        """Let move on to the next minute"""
        next_min = self.trader.nextMin()
        self.raw_sequential_data.append(next_min)
        self.preprocess_df()

        obs = self.get_obs()

        return obs, reward, done, {}

    def reset(self):
        """Reset all values to the default"""

        return self.get_obs()

    def render(self, mode='human', close=False):
        """Render the environment to the screen, by printing on the terminal or creating some graphs"""
