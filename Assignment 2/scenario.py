import os
import pandas as pd
import numpy as np

# Set seed
np.random.seed(42)

cwd = os.path.dirname(__file__)


wind = pd.read_csv(cwd + '/input_data/wind_DK.csv', parse_dates=True)
wind['offshore'] = wind['offshore'] * 200
# Convert time to datetime format
wind['time'] = pd.to_datetime(wind['time'])


prices = pd.read_csv(cwd + '/input_data/Elspotprices.csv', sep=';', decimal=",", parse_dates=True)
prices['HourUTC'] = pd.to_datetime(prices['HourUTC'])
# Reverse the order of the prices
prices = prices.iloc[::-1]


# Function that selectes n random days from the wind data and returns the daily wind production for those days
def wind_scenario_generator(n):
    # Generate n random indices which will be used to select n random days
    indices = np.random.choice(int(len(wind.index)/24), n)
    days = wind.iloc[indices*24]['time'].dt.date
    # Select the n random days of data
    scenario = wind.loc[wind['time'].dt.date.isin(days)]
    return scenario['offshore'].values.reshape(n, 24)

# Function that selectes n random days from the price data and returns the daily prices for those days
def price_scenario_generator(n):
    # Generate n random indices which will be used to select n random days
    indices = np.random.choice(int(len(prices.index)/24), n)
    days = prices.iloc[indices*24]['HourUTC'].dt.date
    # Select the n random days of data
    scenario = prices.loc[prices['HourUTC'].dt.date.isin(days)]
    return scenario['SpotPriceEUR'].values.reshape(n, 24)


# Function that generates n scenarios of 24 binary variables based on a bernoulli distribution with p=0.6
def balance_scenario_generator(n):
    # Generates n scenarios of 24 binary variables based on a bernoulli distribution with p=0.6

    return np.random.choice([0, 1], size=(24, n), p=[0.4, 0.6]).T

wind_scenarios = wind_scenario_generator(20)
price_scenarios = price_scenario_generator(20)
balance_scenarios = balance_scenario_generator(5)
scenarios = [[wind_scenarios[i], price_scenarios[j], balance_scenarios[k]] for i in range(20) for j in range(20) for k in range(5)]
print(np.shape(scenarios))
print(scenarios[0])