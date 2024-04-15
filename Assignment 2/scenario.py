import os
import pandas as pd
import numpy as np

class DataInit:
    def __init__(self):
        # Set seed
        np.random.seed(42)

        cwd = os.path.dirname(__file__)

        self.wind = pd.read_csv(cwd + '/input_data/wind_DK.csv', parse_dates=True)
        self.wind['offshore'] = self.wind['offshore'] * 200
        # Convert time to datetime format
        self.wind['time'] = pd.to_datetime(self.wind['time'])

        self.prices = pd.read_csv(cwd + '/input_data/Elspotprices.csv', sep=';', decimal=",", parse_dates=True)
        self.prices['HourUTC'] = pd.to_datetime(self.prices['HourUTC'])
        # Reverse the order of the prices
        self.prices = self.prices.iloc[::-1]

    def _wind_scenario_generator(self, n):
        indices = np.random.choice(int(len(self.wind.index)/24), n)
        days = self.wind.iloc[indices*24]['time'].dt.date
        scenario = self.wind.loc[self.wind['time'].dt.date.isin(days)]
        return scenario['offshore'].values.reshape(n, 24)

    def _price_scenario_generator(self, n):
        indices = np.random.choice(int(len(self.prices.index)/24), n)
        days = self.prices.iloc[indices*24]['HourUTC'].dt.date
        scenario = self.prices.loc[self.prices['HourUTC'].dt.date.isin(days)]
        return scenario['SpotPriceEUR'].values.reshape(n, 24)

    def _balance_scenario_generator(self, n):
        return np.random.choice([0, 1], size=(24, n), p=[0.4, 0.6]).T

    def generate_scenarios(self, n_wind=20, n_price=20, n_balance=5):
        """
        Generates scenarios for wind, price and balance of format of list of lists of arrays. 
        The length of the outer list is n_wind * n_price * n_balance.
        The length of the inner list is 3,
        where the first element is the wind scenario,
        the second element is the price scenario,
        and the third element is the balance scenario.
        """
        wind_scenarios = self._wind_scenario_generator(n_wind)
        price_scenarios = self._price_scenario_generator(n_price)
        balance_scenarios = self._balance_scenario_generator(n_balance)
        scenarios = [[wind_scenarios[i], price_scenarios[j], balance_scenarios[k]] for i in range(n_wind) for j in range(n_price) for k in range(n_balance)]
        return scenarios
