import gurobipy as gb
from gurobipy import GRB
import numpy as np
import pandas as pd





class Network:
    # Example of reading data from excel, requires openpyxl
    xls = pd.ExcelFile('Assignment 1\data.xlsx')
    df1 = pd.read_excel(xls, 'gen_technical')
    df2 = pd.read_excel(xls, 'gen_cost')

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
    TIMES = ['T1']

    # Offer prices indexed by generator
    c = {'G1':13.32,'G2':13.32,'G3':20.7,'G4':20.93,'G5':26.11,'G6':10.52,
         'G7':10.52,'G8':6.02,'G9':5.47,'G10':0,'G11':10.52,'G12':10.89}
    
    # Bid prices indexed by demand
    u = {'D1':20}

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













































        
