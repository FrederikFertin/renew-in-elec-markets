import gurobipy as gb
from gurobipy import GRB
import numpy as np
import pandas as pd




class Network:
    # Example of reading data from excel, requires openpyxl
    xls = pd.ExcelFile('Assignment 1/data.xlsx')
    gen_tech = pd.read_excel(xls, 'gen_technical')
    gen_econ = pd.read_excel(xls, 'gen_cost')
    system_demand = pd.read_excel(xls, 'demand')
    line_info = pd.read_excel(xls, 'transmission_lines')
    load_info = pd.read_excel(xls, 'demand_nodes')
    wind_tech = pd.read_excel(xls, 'wind_technical')
    wind_profiles = pd.read_csv('Assignment 1/wind_profiles.csv')

    # Number of each type of unit/identity
    G = np.shape(gen_tech)[0]
    D = np.shape(load_info)[0]
    T = np.shape(system_demand)[0]
    L = np.shape(line_info)[0]
    W = np.shape(wind_tech)[0]
    # N = 24 # Number of nodes in network

    #List of Generators, Nodes, Windfarm and Batteries
    GENERATORS = ['G{0}'.format(t) for t in range(1, G+1)]
    DEMANDS = ['D{0}'.format(t) for t in range(1, D+1)]
    LINES = ['L{0}'.format(t) for t in range(1, L+1)]
    TIMES = ['T{0}'.format(t) for t in range(1, T+1)]
    WINDTURBINES = ['W{0}'.format(t) for t in range(1, W+1)]
    # NODES = ['N{0}'.format(t) for t in range(1, N)]

    # Conventional Generator Information
    P_G_max = dict(zip(GENERATORS, gen_tech['P_max'])) # Max generation cap.
    C_G_offer = dict(zip(GENERATORS, gen_econ['C'])) # Generator day-ahead offer price
    
    # Demand Information
    P_D_sum = dict(zip(TIMES, system_demand['System_demand']))
    P_D = {}
    for t, key in enumerate(TIMES):
        P_D[key] = dict(zip(DEMANDS, load_info['load_percent']/100*system_demand['System_demand'][t]))

    # Wind Turbine Production
    p_W_max = 200 # Wind farm maximum (MW)
    WT = ['V{0}'.format(v) for v in wind_tech['Profile']]
    chosen_wind_profiles = wind_profiles[WT] 
    P_W = {}
    for t, key in enumerate(TIMES):
        P_W[key] = dict(zip(WINDTURBINES, chosen_wind_profiles.iloc[t,:] * p_W_max))
    
    # Bid prices indexed by demand
    u = {'D1':20}

    # System consumption indexed by time 
    P_D_sum = {'T1': {D1: 0.2*1775.835, D2: 0.8*1775.835},
      'T2': 2517.975,
      'T3': 1669.815,
      'T4': 2517.975,
      'T5': 1590.3,
      'T6': 2464.965,
      'T7': 1563.795,
      'T8': 2464.965,
      'T9': 1563.795,
      'T10': 2623.995,
      'T11': 1590.3,
      'T12': 2650.5,
      'T13': 1961.37,
      'T14': 2650.5,
      'T15': 2279.43,
      'T16': 2544.48,
      'T17': 2517.975,
      'T18': 2411.955,
      'T19': 2544.48,
      'T20': 2199.915,
      'T21': 2544.48,
      'T22': 1934.865,
      'T23': 2517.975,
      'T24': 1669.815}
    
    # Max generation indexed by generator
    P_G_max = {'G1':152,'G2':152,'G3':350,'G4':591, 'G5':60,  'G6':155,
               'G7':155,'G8':400,'G9':400,'G10':300,'G11':310,'G12':350}
    
    # Demand quantities indexed by demand
    P_D = {'D1':P_D_sum['T1']}


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
        self.variables.consumption = {d:self.model.addVar(lb=0,ub=self.P_D[d],name='consumption of demand {0}'.format(d)) for d in self.DEMANDS}
        self.variables.generator_dispatch = {g:self.model.addVar(lb=0,ub=self.P_G_max[g],name='dispatch of generator {0}'.format(g)) for g in self.GENERATORS} 
        
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
        self.data.consumption_values = {d:self.variables.consumption[d].x for d in self.DEMANDS}
        
        # save generator dispatches 
        self.data.generator_dispatch_values = {g:self.variables.generator_dispatch[g].x for g in self.GENERATORS}
        
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
        self.results.profits = {g:(self.c[g] - self.data.lambda_) * self.data.generator_dispatch_values[g] for g in self.GENERATORS}
        
        # calculate utility of suppliers ( (U_D - lambda) * p_D )
        self.results.utilities = {d:(self.u[d] - self.data.lambda_) * self.data.consumption_values[d] for d in self.DEMANDS}
        
        self._display_results()
        
        
    
if __name__ == "__main__":
    ec = EconomicDispatch()
    ec.run()
    ec.calculate_results()













































        
