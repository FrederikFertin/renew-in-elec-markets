import gurobipy as gb
from gurobipy import GRB
import numpy as np
import pandas as pd

class Network:
    #Wind farm maximum
    p_W_max = 300 # MW

    #Dictionary for wind farms
    p_W_data = {}

    #For loop to collect wind_data
    # for i in range(1,7):
    #     df = pd.read_csv('wind_data/wind '+str(i)+'.out')
    #     p_W_data['W'+str(i)] = df['V1'].values[:24] * p_W_max

    #List of Generators, Nodes, Windfarm and Batteries
    GENERATORS = ['G1','G2','G3', 'G4', 'G5', 'G6', 'G7', 'G8', 'G9', 'G10', 'G11', 'G12']
    DEMANDS = ['D1']
    # NODES = ['N1','N2','N3','N4','N5','N6','N7','N8','N9','N10','N11','N12','N13',
    #          'N14','N15','N16','N17','N18','N19','N20','N21','N22','N23','N24']
    # WIND = ['W1', 'W2', 'W3', 'W4', 'W5', 'W6']
    # BATTERY = ['B1','B2','B3']

    # # Number of nodes 
    # N = len(NODES)

    # Number of generators 
    G = len(GENERATORS)

    # # Number of wind turbines 
    # W = len(WIND)

    # Number of Hours
    TIMES = ['T{0}'.format(t) for t in range(1, 25)]

    # Offer prices indexed by generator
    c = {'G1':13.32,'G2':13.32,'G3':20.7,'G4':20.93,'G5':26.11,'G6':10.52,
         'G7':10.52,'G8':6.02,'G9':5.47,'G10':0,'G11':10.52,'G12':10.89}
    
    # Bid prices indexed by demand
    u = 400 # $/MWh

    # System consumption indexed by time 
    P_D_sum = {'T1': 1775.835,
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
    
    # Fraction of system consumption at each node indexed by loads 
    P_D_fraction = np.array([0.038, 0.034, 0.063, 0.026, 0.025, 0.048, 0.044, 0.06,
                              0.061, 0.068, 0.093, 0.068, 0.111, 0.035, 0.117, 0.064, 
                              0.045])
    
    # Max generation indexed by generator
    P_G_max = {'G1':152,'G2':152,'G3':350,'G4':591, 'G5':60,  'G6':155,
               'G7':155,'G8':400,'G9':400,'G10':300,'G11':310,'G12':350}
    
    # Demand quantities indexed by demand
    P_D = {'D1':P_D_sum}
    
    P_R_DW = {k:-v/2 for k,v in P_G_max.items()}
    P_R_UP = {k:v/2 for k,v in P_G_max.items()}


class expando(object):
    '''
        A small class which can have attributes set
    '''
    pass

class EconomicDispatch(Network):
    
    def __init__(self, n_hours, ramping): # initialize class
        # super().__init__(n_samples=n_samples)
        
        self.data = expando() # build data attributes
        self.variables = expando() # build variable attributes
        self.constraints = expando() # build sontraint attributes
        self.results = expando()
        self.TIMES = self.TIMES[:n_hours]
        self.ramping = ramping
        self._build_model() # build gurobi model
    
    def _build_model(self):
        # initialize optimization model
        self.model = gb.Model(name='Economic Dispatch')
        
        # initialize variables 
        self.variables.consumption = {(d,t):self.model.addVar(lb=0,ub=self.P_D[d][t],name='consumption of demand {0}'.format(d)) for d in self.DEMANDS for t in self.TIMES}
        self.variables.generator_dispatch = {(g,t):self.model.addVar(lb=0,ub=self.P_G_max[g],name='dispatch of generator {0}'.format(g)) for g in self.GENERATORS for t in self.TIMES} 
        
        # initialize objective to maximize social welfare
        demand_utility = gb.quicksum(self.u * self.variables.consumption[d,t] for d in self.DEMANDS for t in self.TIMES)
        generator_costs = gb.quicksum(self.c[g] * self.variables.generator_dispatch[g,t] for g in self.GENERATORS for t in self.TIMES)
        objective = demand_utility - generator_costs
        self.model.setObjective(objective, gb.GRB.MAXIMIZE)
        
        # initialize constraints 
        self.constraints.balance_constraint = {t:self.model.addConstr(
                gb.quicksum(self.variables.consumption[d,t] for d in self.DEMANDS) - gb.quicksum(self.variables.generator_dispatch[g,t] for g in self.GENERATORS),
                gb.GRB.EQUAL,
                0, name='Balance equation') for t in self.TIMES}
        
        T = self.TIMES
        if self.ramping:
            self.constraints.ramping_dw = {(g,t):self.model.addConstr(self.variables.generator_dispatch[g,t] - self.variables.generator_dispatch[g,T[n]], 
                                                               gb.GRB.GREATER_EQUAL, 
                                                               self.P_R_DW[g]) for g in self.GENERATORS for n,t in enumerate(self.TIMES[1:])}
            self.constraints.ramping_up = {(g,t):self.model.addConstr(self.variables.generator_dispatch[g,t] - self.variables.generator_dispatch[g,T[n]], 
                                                               gb.GRB.LESS_EQUAL, 
                                                               self.P_R_UP[g]) for g in self.GENERATORS for n,t in enumerate(self.TIMES[1:])}
        
    def _save_data(self):
        # save objective value
        self.data.objective_value = self.model.ObjVal
        
        # save consumption values 
        self.data.consumption_values = {(d,t):self.variables.consumption[d,t].x for d in self.DEMANDS for t in self.TIMES}
        
        # save generator dispatches 
        self.data.generator_dispatch_values = {(g,t):self.variables.generator_dispatch[g,t].x for g in self.GENERATORS for t in self.TIMES}
        
        # save uniform prices lambda 
        self.data.lambda_ = {t:self.constraints.balance_constraint[t].Pi for t in self.TIMES}
        
    def run(self):
        self.model.optimize()
        self._save_data()

    def calculate_results(self):
        # calculate profits of suppliers ( profits = (C_G - lambda) * p_G )
        self.results.profits = {g:sum((self.data.lambda_[t] - self.c[g])  * self.data.generator_dispatch_values[g,t] for t in self.TIMES) for g in self.GENERATORS}
        
        # calculate utility of suppliers ( (U_D - lambda) * p_D )
        self.results.utilities = {d:sum((self.u - self.data.lambda_[t]) * self.data.consumption_values[d,t] for t in self.TIMES) for d in self.DEMANDS}
    
    def display_results(self):
        print()
        print("-------------------   RESULTS  -------------------")
        print("Market clearing prices: " + str(self.data.lambda_))
        print()
        print("Social welfare: " + str(self.data.objective_value))
        print()
        print("Profit of suppliers: ")
        print(self.results.profits)
        print()
        print("Utility of demands: ")
        print(self.results.utilities)
        

if __name__ == "__main__":
    ec = EconomicDispatch(n_hours=24, ramping=False)
    ec.run()
    ec.calculate_results()
    ec.display_results()













































        
