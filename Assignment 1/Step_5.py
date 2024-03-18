import gurobipy as gb
from gurobipy import GRB
import numpy as np
import pandas as pd
from Step_1_2 import Network, CommonMethods, expando, EconomicDispatch

class BalancingMarket (EconomicDispatch, Network, CommonMethods):
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
         self.data.generator_dispatch_values = {(g,t):self.variables.generator_dispatch[g,t] for g in self.GENERATORS for t in self.TIMES}
         self.data.wind_dispatch_values = {(w,t):self.variables.wind_turbines[w,t] for w in self.WINDTURBINES for t in self.TIMES}
        

         self.model.update()

        

        #  # initialize objective to maximize social welfare
        #  demand_utility = gb.quicksum(self.U_D[t][d] * self.variables.consumption[d,t] for d in self.DEMANDS for t in self.TIMES)
        #  generator_costs = gb.quicksum(self.C_G_offer[g] * self.variables.generator_dispatch[g,t] for g in self.GENERATORS for t in self.TIMES)
        #  objective = demand_utility - generator_costs
        #  self.model.setObjective(objective, gb.GRB.MAXIMIZE)

         upregulation = gb.quicksum((self.C_G_offer[g] + 0.1*self.C_G_offer[g])*self.variables.upregulation[g,'T10'] for g in self.GENERATORS)
         downregulation = gb.quicksum((self.C_G_offer[g] - 0.13*self.C_G_offer[g])*self.variables.downregulation[g,'T10'] for g in self.GENERATORS)
         curt_cost = gb.quicksum(self.variables.demand_curt[d,'T10']*400 for d in self.DEMANDS)
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
         balancing_service_expr = gb.quicksum(
                self.variables.demand_curt[d, 'T10'] + self.variables.wind_upregulation[w, 'T10']
                + self.variables.upregulation[g, 'T10'] - self.variables.downregulation[g, 'T10']
                - self.variables.wind_downregulation[w, 'T10']
                for g in self.GENERATORS for w in self.WINDTURBINES for d in self.DEMANDS
            )
         # Import the required data from Step_1_2.py

         self.model.addConstr(
            balancing_service_expr == (
                self.data.generator_dispatch_values['G9', 'T10'] +
                0.1 * self.data.wind_dispatch_values['W1', 'T10'] +
                0.1 * self.data.wind_dispatch_values['W2', 'T10'] -
                0.15 * self.data.wind_dispatch_values['W4', 'T10'] -
                0.15 * self.data.wind_dispatch_values['W6', 'T10']
            )
        )

         
         self.constraints.upregulation = {
            (g,'T10'): self.model.addConstr(self.variables.upregulation[g,'T10'], gb.GRB.LESS_EQUAL, self.P_G_max[g] - self.data.generator_dispatch_values[g,'T10'])
            for g in self.GENERATORS
        }
         
         self.constraints.downregulation = {
            (g,'T10'): self.model.addConstr(self.variables.downregulation[g,'T10'], gb.GRB.LESS_EQUAL, self.data.generator_dispatch_values[g,'T10'])
            for g in self.GENERATORS
        }
         
         self.constraints.curtailment = {
            (d,'T10'): self.model.addConstr(self.variables.demand_curt[d,'T10'], gb.GRB.LESS_EQUAL, self.P_D_sum['T10'])
            for d in self.DEMANDS
        }
         
         self.model.update()
             
    def _save_data(self):
        self.data.objective_value = self.model.objVal

        # Save generator upregulation values for T10
        self.data.generator_upregulation_values = {(g, 'T10'): self.variables.upregulation[g, 'T10'].x for g in self.GENERATORS}

        # Save generator downregulation values for T10
        self.data.generator_downregulation_values = {(g, 'T10'): self.variables.downregulation[g, 'T10'].x for g in self.GENERATORS}

        # Save wind upregulation values for T10
        self.data.wind_upregulation_values = {(w, 'T10'): self.variables.wind_upregulation[w, 'T10'].x for w in self.WINDTURBINES}

        # Save wind downregulation values for T10
        self.data.wind_downregulation_values = {(w, 'T10'): self.variables.wind_downregulation[w, 'T10'].x for w in self.WINDTURBINES}

        # Save demand curtailment values for T10
        self.data.demand_curtailment_values = {(d, 'T10'): self.variables.demand_curt[d, 'T10'].x for d in self.DEMANDS}


    def clear_balancing_market(self):
        if 'T10' in self.TIMES:
        # Outage in generator 9
            self.data.generator_dispatch_values['G9', 'T10'].ub = 0
                
        # Adjust wind farm productions based on forecast deviations
        for w in ['W1', 'W2']:
            self.data.wind_dispatch_values[w, 'T10'].ub *= 1.15  # 15% higher
        for w in ['W4', 'W6']:
            self.data.wind_dispatch_values[w, 'T10'].ub *= 0.9  # 10% lower

        # Re-run the optimization with the updated values
        self.model.optimize()

        # Save generator upregulation values
        self.data.generator_upregulation_values = {(g, t): self.variables.upregulation[g, t].x for g in self.GENERATORS for t in self.TIMES}


        
        
                
    def run(self):
            self.model.optimize()
            self._save_data()

    def display_results(self):
        print("Modified values for Hour 10:")
        for g in self.GENERATORS:
            print(f"Generator Dispatch for {g}:", self.variables.generator_dispatch[g, 'T10'].x)
        print(sum(self.variables.generator_dispatch[g, 'T10'].x for g in self.GENERATORS))
        for w in self.WINDTURBINES:
            print(f"Wind Turbine {w}:", self.variables.wind_turbines[w, 'T10'].x)
        print("Generator Upregulation Values:")
        for g, value in self.data.generator_upregulation_values.items():
            print(f"Generator {g}, Time {'T10'}: {value}")
    #         print("\nBalancing Market Results for Hour 10:")
    #         print("Upward Balancing Prices:")
    #         print(self.results.upward_prices)
    #         print("Downward Balancing Prices:")
    #         print(self.results.downward_prices)
    #         print("Upward Balancing Services:")
    #         print(self.results.upward_service)
    #         print("Downward Balancing Services:")
    #         print(self.results.downward_service)    

  

    

if __name__ == "__main__":
    bm = BalancingMarket(n_hours=24, ramping=False, battery=False, hydrogen=False)
    bm.clear_balancing_market()
    bm.run()
    bm.display_results()





