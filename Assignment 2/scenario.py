import os
import pandas as pd
import numpy as np

class DataInit:
    def __init__(self):
        # Set seed
        np.random.seed(42)

        cwd = os.path.dirname(__file__)

        self.P_max = 200
        self.TIMES = range(24)

        self.wind = pd.read_csv(cwd + '/input_data/wind_DK.csv', parse_dates=True)
        self.wind['offshore'] = self.wind['offshore'] * self.P_max
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
        return np.random.choice([0, 1], size=(n, 24), p=[0.4, 0.6])

    def generate_scenarios(self, n_wind=20, n_price=20, n_balance=3, seed=42):
        """
        Generates scenarios for wind, price and balance of format of list of dicts of arrays.

        train_size is the fraction of the scenarios that should be used for training.

        The length of the outer list is n_wind * n_price * n_balance.
        The size of the dicts is 3,
        where the first element is the wind scenario, 'wind' with 24 elements,
        the second element is the price scenario, 'lambda' with 24 elements,
        and the third element is the balance scenario, 'system_balance' with 24 elements.

        1 is system balance is in excess, 0 is if it is in deficit.
        """

        np.random.seed(seed)

        wind_scenarios = self._wind_scenario_generator(n_wind)
        price_scenarios = self._price_scenario_generator(n_price)
        balance_scenarios = self._balance_scenario_generator(n_balance)
        self.scenarios = [{'wind' : wind_scenarios[i],
                           'lambda' : price_scenarios[j],
                           'system_balance' : balance_scenarios[k]}
                           for i in range(n_wind) for j in range(n_price) for k in range(n_balance)]
        # Shuffle order of scenarios and split into train and test
        np.random.shuffle(self.scenarios)

    def create_train_test_split(self, train_size: int = 250, k: int | None = None):
        if k is None:
            self.train_scenarios = self.scenarios[:train_size]
            self.test_scenarios = self.scenarios[train_size:]
        else:
            self.train_scenarios = self.scenarios[k * train_size:(k + 1) * train_size]
            self.test_scenarios = self.scenarios[:k*train_size]
            if train_size*(k+1) < len(self.scenarios):
                self.test_scenarios.extend(self.scenarios[(k+1)*train_size:])

        self.n_scenarios = len(self.train_scenarios)
        self.SCENARIOS = range(self.n_scenarios)
        self.pi = np.ones(self.n_scenarios)/self.n_scenarios

        self.lambda_DA = np.array([scenario['lambda'] for scenario in self.train_scenarios]).T
        self.p_real = np.array([scenario['wind'] for scenario in self.train_scenarios]).T
        self.imbalance_direction = np.array([scenario['system_balance'] for scenario in self.train_scenarios]).T

if __name__ == '__main__':
    data = DataInit()
    data.generate_scenarios()
    data.create_train_test_split()