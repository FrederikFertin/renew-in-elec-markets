import gurobipy as gb
from gurobipy import GRB
import numpy as np
import pandas as pd
from Step_1_2 import Network, expando, EconomicDispatch
from network_plots import createNetwork, drawNormal, drawSingleStep, drawLMP


class BalancingMarket (EconomicDispatch):
    def __init__(self, n_hours: int, ramping: bool,battery: bool, hydrogen: bool):
        super().__init__(n_hours, ramping, battery, hydrogen)

        self.data = expando() # build data attributes
        self.variables = expando() # build variable attributes
        self.constraints = expando() # build sontraint attributes
        self.results = expando()
        self.TIMES = self.TIMES[:n_hours]
        self._build_model() # build gurobi model
    
    def _build_model(self):
         # initialize optimization model
         self.model = gb.Model(name='Balancing Market')

            # Create variables
         self.variables.consumption = {(d,t):self.model.addVar(lb=0,ub=self.P_D[t][d],name='consumption of demand {0}'.format(d)) for d in self.DEMANDS for t in self.TIMES}
         self.variables.generator_dispatch = {(g,t):self.model.addVar(lb=0,ub=self.P_G_max[g],name='dispatch of generator {0}'.format(g)) for g in self.GENERATORS for t in self.TIMES}
         self.variables.wind_turbines = {(w,t):self.model.addVar(lb=0,ub=self.P_W[t][w],name='dispatch of wind turbine {0}'.format(w)) for w in self.WINDTURBINES for t in self.TIMES}
         self.variables.upregulation = {(g,t):self.model.addVar(lb=0,name='upregulation of generator {0}'.format(g)) for g in self.GENERATORS for t in self.TIMES}
         self.variables.downregulation = {(g,t):self.model.addVar(lb=0,name='downregulation of generator {0}'.format(g)) for g in self.GENERATORS for t in self.TIMES}
         self.variables.wind_upregulation = {(w,t):self.model.addVar(lb=0,name='upregulation of wind turbine {0}'.format(w)) for w in self.WINDTURBINES for t in self.TIMES}
         self.variables.wind_downregulation = {(w,t):self.model.addVar(lb=0,name='downregulation of wind turbine {0}'.format(w)) for w in self.WINDTURBINES for t in self.TIMES}
         self.variables.demand_curt = {(d,t):self.model.addVar(lb=0,name='curtailment of demand {0}'.format(d)) for d in self.DEMANDS for t in self.TIMES}
         #self.variables.wind_curt = {(w,t):self.model.addVar(lb=0,name='curtailment of wind turbine {0}'.format(w)) for w in self.WINDTURBINES for t in self.TIMES}


         self.model.update()


        

        #  # initialize objective to maximize social welfare
        #  demand_utility = gb.quicksum(self.U_D[t][d] * self.variables.consumption[d,t] for d in self.DEMANDS for t in self.TIMES)
        #  generator_costs = gb.quicksum(self.C_G_offer[g] * self.variables.generator_dispatch[g,t] for g in self.GENERATORS for t in self.TIMES)
        #  objective = demand_utility - generator_costs
        #  self.model.setObjective(objective, gb.GRB.MAXIMIZE)

         upregulation = gb.quicksum((self.C_G_offer[g] + 0.1*self.C_G_offer[g])*self.variables.upregulation[g,t] for g in self.GENERATORS for t in self.TIMES)
         downregulation = gb.quicksum((self.C_G_offer[g] - 0.12*self.C_G_offer[g])*self.variables.downregulation[g,t] for g in self.GENERATORS for t in self.TIMES)
         curt_cost = gb.quicksum(self.variables.demand_curt[d,10]*400 for d in self.DEMANDS for t in self.TIMES)
         objective = upregulation + curt_cost - downregulation
         self.model.setObjective(objective, gb.GRB.MINIMIZE)

        # Set objective function (sum(generator dispatch prices for T10))
         #Sum af (generator_dispatch_values prices for T10 + 0.1* prices of generator T10)*upregulation pr. generator + curt_cost*demand_curt
         # - (Generator_dispatch_values prices for T10 - 0.12* prices of generator T10)*downregulation pr. generator
         #objective_expr = gb.quicksum((self.C_G_offer[g] + 0.1*self.C_G_offer[g]) * self.variables.generator_dispatch[g,10] for g in self.GENERATORS for t in self.TIMES)
        # objective_expr += gb.quicksum(self.variables.wind_dispatch[(w, t)] for w, t in self.variables.wind_dispatch)
        # self.model.setObjective(objective_expr, gb.GRB.MINIMIZE)

        #self.model.optimize()
         
         #initialize constraints
         self.constraints.balancing_service = {
            (t): self.model.addConstr(
                self.variables.demand_curt[d,10] + self.variables.upregulation_wind[w,10] 
                + self.variables.upregulation[g,10] - self.variables.downregulation[g,10] 
                - self.variables.downregulation_wind[w,10], gb.GRB.EQUAL, 0
            ) 
            for t in self.TIMES
        }
         
         self.constraints.upregulation = {
            (g,t): self.model.addConstr(self.variables.upregulation[g,10], gb.GRB.LESS_EQUAL, self.P_G_max[g] - self.variables.generator_dispatch[g,t])
            for g in self.GENERATORS for t in self.TIMES
        }
         
         self.constraints.downregulation = {
            (g,t): self.model.addConstr(self.variables.downregulation[g,10], gb.GRB.LESS_EQUAL, self.variables.generator_dispatch[g,10])
            for g in self.GENERATORS for t in self.TIMES
        }
         
         self.constraints.curtailment = {
            (d,t): self.model.addConstr(self.variables.demand_curt[d,10], gb.GRB.LESS_EQUAL, self.P_D_sum[10])
            for d in self.DEMANDS for t in self.TIMES
        }
             
    def _save_data(self):
        #Save objective value
        self.data.objective_value = self.model.objVal

        #Save generator upregulation values
        self.data.generator_upregulation_values = {(g, t): self.variables.upregulation[g, t].x for g in self.GENERATORS for t in self.TIMES}

        #Save generator downregulation values
        self.data.generator_downregulation_values = {(g, t): self.variables.downregulation[g, t].x for g in self.GENERATORS for t in self.TIMES}

        #Save wind upregulation values
        self.data.wind_upregulation_values = {(w, t): self.variables.wind_upregulation[w, t].x for w in self.WINDTURBINES for t in self.TIMES}

        #Save wind downregulation values
        self.data.wind_downregulation_values = {(w, t): self.variables.wind_downregulation[w, t].x for w in self.WINDTURBINES for t in self.TIMES}

        #Save demand curtailment values
        self.data.demand_curtailment_values = {(d, t): self.variables.demand_curt[d, t].x for d in self.DEMANDS for t in self.TIMES}

        



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




