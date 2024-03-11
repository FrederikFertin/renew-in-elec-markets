import gurobipy as gb
from gurobipy import GRB
import numpy as np
import pandas as pd
from Step_1_2 import Network, expando, EconomicDispatch
from network_plots import createNetwork, drawNormal, drawSingleStep, drawLMP


class BalancingMarket (EconomicDispatch):
    def __init__(self, n_hours: int, ramping: bool, battery: bool, hydrogen: bool):
        super().__init__(n_hours, ramping, battery, hydrogen)

        self.data = expando() # build data attributes
        self.variables = expando() # build variable attributes
        self.constraints = expando() # build sontraint attributes
        self.results = expando()
        self._balancing_model() # build gurobi model
    
    def _balancing_model(self):
         # initialize optimization model
         self.model = gb.Model(name='Balancing Market')



        
    #     self.variables.generator_dispatch = {}
    #     self.variables.wind_dispatch = {}
    #     for g, t in self.data.generator_dispatch_values_modified:
    #         self.variables.generator_dispatch[(g, t)] = self.model.addVar(
    #             lb=0, ub=self.data.generator_dispatch_values_modified[(g, t)],
    #             name=f'GeneratorDispatch_{g}_{t}')

    #     for w, t in self.data.wind_dispatch_values_modified:
    #         self.variables.wind_dispatch[(w, t)] = self.model.addVar(
    #             lb=0, ub=self.data.wind_dispatch_values_modified[(w, t)],
    #             name=f'WindDispatch_{w}_{t}')
            
    #     self.model.update()

        # Set objective function (sum(generator dispatch prices for T10))
        # objective_expr = gb.quicksum(self.variables.generator_dispatch[(g, t)] for g, t in self.variables.generator_dispatch)
        # objective_expr += gb.quicksum(self.variables.wind_dispatch[(w, t)] for w, t in self.variables.wind_dispatch)
        # self.model.setObjective(objective_expr, gb.GRB.MINIMIZE)

        # Add constraints
        # Define and add your constraints here

        #self.model.optimize()
        


    def calculate_results(self):
         # To store modified values
        self.data.generator_dispatch_values_modified = {}
        self.data.wind_dispatch_values_modified = {}
        
        #Gen 9 failure
        for g, t in self.data.generator_dispatch_values:
            if g == 'G9':
                self.data.generator_dispatch_values_modified[(g, t)] = 0
            else:
                self.data.generator_dispatch_values_modified[(g, t)] = self.data.generator_dispatch_values[(g, t)]
        
        #Wind production changes
        for w, t in self.data.wind_dispatch_values:
            if w in ['W1', 'W2']:
                self.data.wind_dispatch_values_modified[(w, t)] = 1.15 * self.data.wind_dispatch_values[(w, t)]
            elif w in ['W4', 'W6']:
                self.data.wind_dispatch_values_modified[(w, t)] = 0.9 * self.data.wind_dispatch_values[(w, t)]
            else:
                self.data.wind_dispatch_values_modified[(w, t)] = self.data.wind_dispatch_values[(w, t)]

        
        super().calculate_results()

    def display_modified_values(self, hour = 'T10'):
        print(f"Modified Generator Dispatch Values for hour {hour}:")
        for g, t in self.data.generator_dispatch_values_modified:
            if t == hour:
                print(f"{g}: {self.data.generator_dispatch_values_modified[(g, t)]}")
        
        print(f"\nModified Wind Dispatch Values for hour {hour}:")
        for w, t in self.data.wind_dispatch_values_modified:
            if t == hour:
                print(f"{w}: {self.data.wind_dispatch_values_modified[(w, t)]}")

 #Set objetive function and constraints. 


        

if __name__ == "__main__":
    bm = BalancingMarket(n_hours=24, ramping=False, battery=False, hydrogen=False)
    bm.run()
    bm.calculate_results()
    bm.display_results()
    bm.display_modified_values()




