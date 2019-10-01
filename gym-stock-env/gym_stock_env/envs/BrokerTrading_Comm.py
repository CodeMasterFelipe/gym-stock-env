"""
will have the implementation with Alpaca trader api.
for the moment it will take a data frame and emulate the communication with the api for training purposes

Also Polygon API connection is required for real time data: alpaca gives it for free

"""
import collections

import pandas as pd


class StockMarket_Comm:

    def __init__(self, df):
        """


        :param df:
        ["Unix Timestamp", "Open", "High", "Low", "Close","Volume"]
        """
        self.balance = 10000
        self.num_of_shares = 0
        self.buying_power_multiplier = 4

        self.df = df
        """ need to change this to df"""
        self.open_Col = df['Open'].values.tolist()

        self.current_index = 0

    def getPastData(self, num_of_min):
        """
        should only be call at the start!
        :param num_of_min: how minutes on the past should it return data
        :return: It returns a data frame (pandas) from num_of_min in the past to the most recent 'close' min
        """
        prev_minutes = self.df.head(num_of_min)

        self.current_index = num_of_min

        return prev_minutes.values.tolist()

    def nextMin(self):
        """
        this only work on training data. it will move current minute to +1, updating 'present'
        :return: It return the previous min, with the closing price
        """
        last_minute = self.df.iloc[self.current_index].values.tolist()

        self.current_index += 1  # moving to one min in the future

        return last_minute

    def getCurrentPrice(self):
        price = self.df.at[self.current_index, "Open"]
        print(price)
        return price

    def buy(self, shares):
        # check if there is balance to buy them
        stock_price = self.getCurrentPrice()
        cost = stock_price * shares

        if cost > self.balance:
            print("can't process, cost is higher than the balance")
            return -1

        self.num_of_shares += shares
        self.balance -= cost

        return 1

    def buyMax(self, use_buying_power):

        stock_price = self.getCurrentPrice()
        shares = 0
        if use_buying_power:
            shares = int((self.balance * self.buying_power_multiplier) / stock_price)
        else:
            share = int(self.balance / stock_price)

        cost = shares * stock_price

        self.num_of_shares += shares
        self.balance -= cost

        return 1

    def sellALl(self):

        stock_price = self.getCurrentPrice()

        if self.num_of_shares == 0:
            print("Trying to sell when there is no shares to sell")
            return -1

        selling_price = stock_price * self.num_of_shares

        self.balance = selling_price
        self.num_of_shares = 0

        return 1

    def getAccountInfo(self):
        return self.balance, self.num_of_shares

    def distanceFromLocalMin(self):
        current_price = self.getCurrentPrice()

        # check to the left of the graph first
        index = self.current_index
        lowest_price = current_price
        lowest_index = index
        found_low = False
        while index > 0:
            index -= 1
            price = self.open_Col[index]
            if price < current_price:
                lowest_price = price
                lowest_index = index - 1
                found_low = True
            else:
                break

        if not found_low:
            index = self.current_index
            while index < len(self.open_Col):
                index += 1
                price = self.open_Col[index]
                if price < current_price:
                    lowest_price = price
                    lowest_index = index + 1
                    found_low = True
                else:
                    break

        lost = 0

        if found_low:
            lost = (lowest_index - index) ** 2

        return lost

    def distanceFromLocalMax(self):
        current_price = self.getCurrentPrice()

        # check to the left of the graph first
        index = self.current_index
        highest_price = current_price
        highest_index = index
        found_high = False
        while index > 0:
            index -= 1
            price = self.open_Col[index]
            if price > current_price:
                highest_price = price
                highest_index = index - 1
                found_high = True
            else:
                break

        if not found_high:
            index = self.current_index
            while index < len(self.open_Col):
                index += 1
                price = self.open_Col[index]
                if price > current_price:
                    highest_price = price
                    highest_index = index + 1
                    found_high = True
                else:
                    break

        lost = 0

        if found_high:
            lost = (highest_index - index) ** 2

        return lost

    def broke(self):
        if self.balance < 0:
            if self.num_of_shares == 0:
                return True

        return False
