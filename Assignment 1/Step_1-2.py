import gurobipy as gb
from gurobipy import GRB
import numpy as np
import pandas as pd




class Network:
    # Reading data from Excel, requires openpyxl
    xls = pd.ExcelFile('Assignment 1/data.xlsx')
    gen_tech = pd.read_excel(xls, 'gen_technical')
    gen_econ = pd.read_excel(xls, 'gen_cost')
    system_demand = pd.read_excel(xls, 'demand')
    line_info = pd.read_excel(xls, 'transmission_lines')
    load_info = pd.read_excel(xls, 'demand_nodes')
    wind_tech = pd.read_excel(xls, 'wind_technical')

    # Loading csv file of normalized wind profiles
    wind_profiles = pd.read_csv('Assignment 1/wind_profiles.csv')

    # Number of each type of unit/identity
    G = np.shape(gen_tech)[0] # Number of generators
    D = np.shape(load_info)[0] # Number of loads/demands
    T = np.shape(system_demand)[0] # Number of time periods/hours
    L = np.shape(line_info)[0] # Number of transmission lines
    W = np.shape(wind_tech)[0] # Number of wind farms
    # N = 24 # Number of nodes in network

    # Lists of Generators etc.
    GENERATORS = ['G{0}'.format(t) for t in range(1, G+1)]
    DEMANDS = ['D{0}'.format(t) for t in range(1, D+1)]
    LINES = ['L{0}'.format(t) for t in range(1, L+1)]
    TIMES = ['T{0}'.format(t) for t in range(1, T+1)]
    WINDTURBINES = ['W{0}'.format(t) for t in range(1, W+1)]
    # NODES = ['N{0}'.format(t) for t in range(1, N)]

    ## Conventional Generator Information
    P_G_max = dict(zip(GENERATORS, gen_tech['P_max'])) # Max generation cap.
    C_G_offer = dict(zip(GENERATORS, gen_econ['C'])) # Generator day-ahead offer price
    
    ## Demand Information
    P_D_sum = dict(zip(TIMES, system_demand['System_demand'])) # Total system demands
    P_D = {} # Distribution of system demands
    for t, key in enumerate(TIMES):
        P_D[key] = dict(zip(DEMANDS, load_info['load_percent']/100*system_demand['System_demand'][t]))
    U_D = dict(zip(DEMANDS, load_info['bid_price'])) # Demand bidding price <- set values in excel

    ## Wind Turbine Information
    p_W_cap = 200 # Wind farm capcities (MW)
    WT = ['V{0}'.format(v) for v in wind_tech['Profile']]
    chosen_wind_profiles = wind_profiles[WT] # 'Randomly' chosen profiles for each wind farm
    P_W = {} # Wind production for each hour and each wind farm
    for t, key in enumerate(TIMES):
        P_W[key] = dict(zip(WINDTURBINES, chosen_wind_profiles.iloc[t,:] * p_W_cap))
    


class expando(object):
    '''
        A small class which can have attributes set
    '''
    pass

class EconomicDispatch(Network):
    
    def __init__(self): # initialize class
        # super().__init__(n_samples=n_samples)
        
        self.data = expando() # build data attributes
        self.variables = expando() # build variable attributes
        self.constraints = expando() # build sontraint attributes
        self.results = expando()
        self._build_model() # build gurobi model
    
    def _build_model(self):
        # initialize optimization model
        self.model = gb.Model(name='Economic Dispatch')
        
        # initialize variables 
        self.variables.consumption = {d : self.model.addVar(lb=0,ub=self.P_D[d], name='consumption of demand {0}'.format(d)) for d in self.DEMANDS}
        self.variables.generator_dispatch = {g : self.model.addVar(lb=0,ub=self.P_G_max[g], name='dispatch of generator {0}'.format(g)) for g in self.GENERATORS} 
        
        # initialize objective to maximize social welfare
        demand_utility = gb.quicksum(self.u[d] * self.variables.consumption[d] for d in self.DEMANDS)
        generator_costs = gb.quicksum(self.c[g] * self.variables.generator_dispatch[g] for g in self.GENERATORS)
        objective = demand_utility - generator_costs
        self.model.setObjective(objective, gb.GRB.MAXIMIZE)
        
        # initialize constraints
        self.constraints.balance_constraint = self.model.addConstr(
                gb.quicksum(self.variables.generator_dispatch[g] for g in self.GENERATORS),
                gb.GRB.EQUAL,
                gb.quicksum(self.variables.consumption[d] for d in self.DEMANDS),name='Balance equation') # day-ahead balance equation
        
    def _save_data(self):
        # save objective value
        self.data.objective_value = self.model.ObjVal
        
        # save consumption values 
        self.data.consumption_values = {d : self.variables.consumption[d].x for d in self.DEMANDS}
        
        # save generator dispatches 
        self.data.generator_dispatch_values = {g : self.variables.generator_dispatch[g].x for g in self.GENERATORS}
        
        # save uniform prices lambda 
        self.data.lambda_ = self.constraints.balance_constraint.Pi
        
    def run(self):
        self.model.optimize()
        self._save_data()

    def _display_results(self):
        print()
        print("-------------------   RESULTS  -------------------")
        print("Market clearing price: " + str(np.round(self.data.lambda_, decimals=2)))
        print()
        print("Social welfare: " + str(self.data.objective_value))
        print()
        print("Profit of suppliers: ")
        print(self.results.profits)
        print()
        print("Utility of demands: ")
        print(self.results.utilities)

    def calculate_results(self):
        # calculate profits of suppliers ( profits = (C_G - lambda) * p_G )
        self.results.profits = {g : (self.c[g] - self.data.lambda_) * self.data.generator_dispatch_values[g] for g in self.GENERATORS}
        
        # calculate utility of suppliers ( (U_D - lambda) * p_D )
        self.results.utilities = {d : (self.u[d] - self.data.lambda_) * self.data.consumption_values[d] for d in self.DEMANDS}
        
        self._display_results()
        
        
    
if __name__ == "__main__":
    ec = EconomicDispatch()
    ec.run()
    ec.calculate_results()













































        
