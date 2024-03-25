import gurobipy as gb
from gurobipy import GRB
import numpy as np
import pandas as pd
from Step_1_2 import expando, EconomicDispatch

class BalancingMarket(EconomicDispatch):
    def __init__(self, ramping: bool, battery: bool, hydrogen: bool, hour: str):
        super().__init__(n_hours=24, ramping=ramping, battery=battery, hydrogen=hydrogen)

        self.bm_data = expando() # build data attributes
        self.bm_variables = expando() # build variable attributes
        self.bm_constraints = expando() # build contraint attributes
        self.bm_results = expando() # build
        self.hour = hour

    def build_bm_model(self):
         # initialize optimization model
         self.bm_model = gb.Model(name='Balancing Market')

         # Create variables
         #self.bm_variables.consumption = {(d,t):self.model.addVar(lb=0,ub=self.P_D[t][d],name='consumption of demand {0}'.format(d)) for d in self.DEMANDS for t in self.TIMES}
         #self.bm_variables.generator_dispatch = {(g,t):self.model.addVar(lb=0, ub=self.P_G_max[g],name='dispatch of generator {0}'.format(g)) for g in self.GENERATORS for t in self.TIMES}
         #self.bm_variables.wind_turbines = {(w,t):self.model.addVar(lb=0, ub=self.P_W[t][w],name='dispatch of wind turbine {0}'.format(w)) for w in self.WINDTURBINES for t in self.TIMES}
         self.bm_variables.upregulation = {g: self.bm_model.addVar(lb=0, name='upregulation of generator {0}'.format(g)) for g in self.GENERATORS}
         self.bm_variables.downregulation = {g: self.bm_model.addVar(lb=0, name='downregulation of generator {0}'.format(g)) for g in self.GENERATORS}
         self.bm_variables.wind_upregulation = {w: self.bm_model.addVar(lb=0, name='upregulation of wind turbine {0}'.format(w)) for w in self.WINDTURBINES}
         self.bm_variables.wind_downregulation = {w: self.bm_model.addVar(lb=0, name='downregulation of wind turbine {0}'.format(w)) for w in self.WINDTURBINES}
         self.bm_variables.demand_curt = self.bm_model.addVar(lb=0, name='curtailment of demand')
         #self.bm_variables.wind_curt = {(w,t):self.model.addVar(lb=0,name='curtailment of wind turbine {0}'.format(w)) for w in self.WINDTURBINES for t in self.TIMES}

         self.bm_model.update()

         #  # initialize objective to maximize social welfare
         #  demand_utility = gb.quicksum(self.U_D[t][d] * self.bm_variables.consumption[d,t] for d in self.DEMANDS for t in self.TIMES)
         #  generator_costs = gb.quicksum(self.C_G_offer[g] * self.bm_variables.generator_dispatch[g,t] for g in self.GENERATORS for t in self.TIMES)
         #  objective = demand_utility - generator_costs
         #  self.model.setObjective(objective, gb.GRB.MAXIMIZE)

         upregulation_cost = gb.quicksum((self.data.lambda_[self.hour] + 0.1*self.C_G_offer[g])*self.bm_variables.upregulation[g] for g in self.GENERATORS)
         downregulation_cost = gb.quicksum((self.data.lambda_[self.hour] - 0.13*self.C_G_offer[g])*self.bm_variables.downregulation[g] for g in self.GENERATORS)
         curt_cost = self.bm_variables.demand_curt*self.U_D_curt
         objective = curt_cost + upregulation_cost - downregulation_cost
         self.bm_model.setObjective(objective, gb.GRB.MINIMIZE)

         # Set objective function (sum(generator dispatch prices for T10))
         #Sum af (generator_dispatch_values prices for T10 + 0.1* prices of generator T10)*upregulation pr. generator + curt_cost*demand_curt
         # - (Generator_dispatch_values prices for T10 - 0.12* prices of generator T10)*downregulation pr. generator
         #objective_expr = gb.quicksum((self.C_G_offer[g] + 0.1*self.C_G_offer[g]) * self.variables.generator_dispatch[g,10] for g in self.GENERATORS for t in self.TIMES)
         # objective_expr += gb.quicksum(self.variables.wind_dispatch[(w, t)] for w, t in self.variables.wind_dispatch)
         # self.model.setObjective(objective_expr, gb.GRB.MINIMIZE)

         #self.model.optimize()
         
         #initialize constraints
         balancing_service_lhs = gb.quicksum(
                self.bm_variables.demand_curt
                + self.bm_variables.upregulation[g]
                - self.bm_variables.downregulation[g]
                for g in self.GENERATORS
         )
         balancing_service_rhs = (
                self.data.generator_dispatch_values['G9', self.hour] +
                0.1 * self.data.wind_dispatch_values['W1', self.hour] +
                0.1 * self.data.wind_dispatch_values['W2', self.hour] -
                0.15 * self.data.wind_dispatch_values['W4', self.hour] -
                0.15 * self.data.wind_dispatch_values['W6', self.hour]
         )
         self.power_deficit = (balancing_service_rhs > 0)

         self.bm_constraints.bm_balance = self.bm_model.addConstr(
            balancing_service_lhs,
            gb.GRB.EQUAL,
            balancing_service_rhs
         )

         
         self.bm_constraints.upregulation = {
            g: self.bm_model.addConstr(
                self.bm_variables.upregulation[g],
                gb.GRB.LESS_EQUAL,
                ((self.P_G_max[g] - self.data.generator_dispatch_values[g,self.hour]) if g != 'G9' else 0)
            ) for g in self.GENERATORS
         }
         
         self.bm_constraints.downregulation = {
            g: self.bm_model.addConstr(
                self.bm_variables.downregulation[g],
                gb.GRB.LESS_EQUAL,
                (self.data.generator_dispatch_values[g,self.hour] if g != 'G9' else 0)
            ) for g in self.GENERATORS
         }

         self.bm_constraints.curtailment = self.bm_model.addConstr(
             self.bm_variables.demand_curt, gb.GRB.LESS_EQUAL, self.P_D_sum[self.hour]
         )

         self.bm_model.update()

    def _save_bm_data(self):
        # Save objective value
        self.bm_data.objective_value = self.bm_model.objVal

        # Save generator up- and downregulation values
        self.bm_data.upregulation_values = {g: self.bm_variables.upregulation[g].x for g in self.GENERATORS}
        self.bm_data.downregulation_values = {g: self.bm_variables.downregulation[g].x for g in self.GENERATORS}

        # Save bm price
        self.bm_data.lambda_ = self.bm_constraints.bm_balance.Pi

        # Save demand curtailment values for T10
        self.bm_data.demand_curtailment_values = self.bm_variables.demand_curt.x

    def clear_bm(self):
        """
        if self.hour in self.TIMES:
        # Outage in generator 9
            self.data.generator_dispatch_values['G9', self.hour].ub = 0
                
        # Adjust wind farm productions based on forecast deviations
        for w in ['W1', 'W2']:
            self.data.wind_dispatch_values[w, self.hour].ub *= 1.15  # 15% higher
        for w in ['W4', 'W6']:
            self.data.wind_dispatch_values[w, self.hour].ub *= 0.9  # 10% lower
        """
        # Re-run the optimization with the updated values
        self.bm_model.optimize()
        self._save_bm_data()

    def calculate_bm_results(self, pricing_scheme: str):
        if pricing_scheme == 'one-price':
            self.bm_results.profits_G = {g:
                (self.bm_data.lambda_ - self.C_G_offer[g]) * (self.bm_data.upregulation_values[g] - self.bm_data.downregulation_values[g])
                for g in self.GENERATORS
            }
            self.bm_results.profits_G['G9'] = (self.C_G_offer['G9'] - self.bm_data.lambda_) * self.data.generator_dispatch_values['G9', self.hour]
            self.bm_results.profits_W = {w:
                - self.bm_data.lambda_ * 0.1 * self.data.wind_dispatch_values[w, self.hour] for w in ['W1', 'W2']
            }
            for w in ['W4', 'W6']:
                self.bm_results.profits_W[w] = self.bm_data.lambda_ * 0.15 * self.data.wind_dispatch_values[w, self.hour]

        elif pricing_scheme == 'two-price':
            self.bm_results.profits_G = {g:
                self.bm_data.lambda_ * (self.bm_data.upregulation_values[g] - self.bm_data.downregulation_values[g])
                for g in self.GENERATORS
            }
            if self.power_deficit:
                self.bm_results.profits_G['G9'] = (self.C_G_offer['G9'] - self.bm_data.lambda_) * self.data.generator_dispatch_values['G9', self.hour]
                self.bm_results.profits_W = {w:
                    - self.bm_data.lambda_ * 0.1 * self.data.wind_dispatch_values[w, self.hour] for w in ['W1', 'W2']
                }
                for w in ['W4', 'W6']:
                    self.bm_results.profits_W[w] = self.data.lambda_[self.hour] * 0.15 * self.data.wind_dispatch_values[w, self.hour]
            else:
                self.bm_results.profits_G['G9'] = (self.C_G_offer['G9'] - self.data.lambda_[self.hour]) * self.data.generator_dispatch_values[
                    'G9', self.hour]
                self.bm_results.profits_W = {w:
                    - self.data.lambda_[self.hour] * 0.1 * self.data.wind_dispatch_values[w, self.hour] for w in ['W1', 'W2']
                }
                for w in ['W4', 'W6']:
                    self.bm_results.profits_W[w] = self.bm_data.lambda_ * 0.15 * self.data.wind_dispatch_values[w, self.hour]

        else:
            raise NotImplementedError

    def calculate_DA_results(self):
        # calculate profits of suppliers ( profits = (C_G - lambda) * p_G )
        self.results.profits_G = {g:
            (self.data.lambda_[self.hour] - self.C_G_offer[g]) * self.data.generator_dispatch_values[g, self.hour]
                                  for g in self.GENERATORS}
        self.results.profits_W = {
            w: self.data.lambda_[self.hour] * self.data.wind_dispatch_values[w, self.hour] for w in
            self.WINDTURBINES}

        # calculate utility of suppliers ( (U_D - lambda) * p_D )
        self.results.utilities = {d:
            (self.U_D[self.hour][d] - self.data.lambda_[self.hour]) * self.data.consumption_values[d, self.hour]
            for d in self.DEMANDS}

    def display_results(self):
        print("Modified values for Hour 10:")
        for g in self.GENERATORS:
            print(f"Generator Dispatch for {g}:", self.variables.generator_dispatch[g, self.hour].x)
        print(sum(self.variables.generator_dispatch[g, self.hour].x for g in self.GENERATORS))
        for w in self.WINDTURBINES:
            print(f"Wind Turbine {w}:", self.variables.wind_turbines[w, self.hour].x)
        print("Generator Upregulation Values:")
        for g, value in self.data.generator_upregulation_values.items():
            print(f"Generator {g}, Time {self.hour}: {value}")
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
    bm = BalancingMarket(ramping=False, battery=False, hydrogen=False, hour='T10')
    # Clear DA market
    bm.run()
    # Calculate DA results
    bm.calculate_DA_results()
    # Now we clear the BM
    bm.build_bm_model()
    bm.clear_bm()
    bm.calculate_bm_results(pricing_scheme='one-price')
    bm.calculate_bm_results(pricing_scheme='two-price')
    bm.display_results()





