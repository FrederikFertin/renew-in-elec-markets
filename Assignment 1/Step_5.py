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
        self.bm_constraints = expando() # build constraint attributes
        self.bm_results = expando() # build result attributes
        self.hour = hour # hour for which the balancing market is being cleared

    def build_bm_model(self):
        """ Assumes that the day-ahead market has been cleared and the data is available """

        ## Initialize optimization model
        self.bm_model = gb.Model(name='Balancing Market')

        ## Create variables - only conventional generators are participating in the balancing market
        self.bm_variables.upregulation = {g: self.bm_model.addVar(lb=0, name='upregulation of generator {0}'.format(g)) for g in self.GENERATORS}
        self.bm_variables.downregulation = {g: self.bm_model.addVar(lb=0, name='downregulation of generator {0}'.format(g)) for g in self.GENERATORS}
        self.bm_variables.demand_curt = self.bm_model.addVar(lb=0, name='curtailment of demand')

        self.bm_model.update()

        ## Set objective function - minimize system balancing costs
        upregulation_cost = gb.quicksum((self.data.lambda_[self.hour] + 0.1*self.C_G_offer[g])*self.bm_variables.upregulation[g] for g in self.GENERATORS)
        downregulation_cost = gb.quicksum((self.data.lambda_[self.hour] - 0.13*self.C_G_offer[g])*self.bm_variables.downregulation[g] for g in self.GENERATORS)
        curt_cost = self.bm_variables.demand_curt*self.U_D_curt
        objective = curt_cost + upregulation_cost - downregulation_cost
        self.bm_model.setObjective(objective, gb.GRB.MINIMIZE)

        ## Initialize constraints
        # Generator and demand balancing response
        balancing_service_lhs = gb.quicksum(
                self.bm_variables.demand_curt
                + self.bm_variables.upregulation[g]
                - self.bm_variables.downregulation[g]
                for g in self.GENERATORS
        )

        # Imbalance in the system
        balancing_service_rhs = (
                self.data.generator_dispatch_values['G9', self.hour] +
                0.1 * self.data.wind_dispatch_values['W1', self.hour] +
                0.1 * self.data.wind_dispatch_values['W2', self.hour] -
                0.15 * self.data.wind_dispatch_values['W4', self.hour] -
                0.15 * self.data.wind_dispatch_values['W6', self.hour]
        )

        # Check if there is a power deficit or surplus
        self.power_deficit = (balancing_service_rhs > 0)

        # Balance balancing services with imbalance in the system
        self.bm_constraints.bm_balance = self.bm_model.addConstr(
            balancing_service_lhs,
            gb.GRB.EQUAL,
            balancing_service_rhs
        )

        # The upregulation cannot exceed the additional available capacity for a generator (G9 is in failure mode)
        self.bm_constraints.upregulation = {
            g: self.bm_model.addConstr(
                self.bm_variables.upregulation[g],
                gb.GRB.LESS_EQUAL,
                ((self.P_G_max[g] - self.data.generator_dispatch_values[g,self.hour]) if g != 'G9' else 0)
            ) for g in self.GENERATORS
        }

        # The downregulation cannot be greater than its planned production for a generator (G9 is in failure mode)
        self.bm_constraints.downregulation = {
            g: self.bm_model.addConstr(
                self.bm_variables.downregulation[g],
                gb.GRB.LESS_EQUAL,
                (self.data.generator_dispatch_values[g,self.hour] if g != 'G9' else 0)
            ) for g in self.GENERATORS
        }

        # Load curtailment cannot exceed the total demand
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
        # Re-run the optimization with the updated values
        self.bm_model.optimize()
        self._save_bm_data()

    def calculate_bm_results(self, pricing_scheme: str):
        """ Calculate profits for generators and wind turbines in one-price and two-price schemes """

        ## One-price scheme uses the balancing market price for all generators and wind turbines
        ## for both up- and down-regulation
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

        ## Two-price scheme uses the balancing market price for units creating imbalance in the system,
        ## the day-ahead market price for units that are involuntarily balancing the system,
        ## and the balancing price for units that are providing balancing services.
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
        """ Calculate DA profits for the specific hour """
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
        print()
        print("-------------------   RESULTS  -------------------")
        print("Balancing price: " + str(self.bm_data.lambda_))
        print()
        print("Profit of suppliers: ")
        print("Generators:")
        print(self.bm_results.profits_G)
        print("Wind turbines:")
        print(self.bm_results.profits_W)

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
    bm.display_results()
