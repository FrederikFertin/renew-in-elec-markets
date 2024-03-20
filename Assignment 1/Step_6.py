import gurobipy as gb
from gurobipy import GRB
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from Step_1_2 import Network, expando
from Step_2 import CommonMethods
# from network_plots import createNetwork, drawNormal, drawSingleStep, drawLMP




class ReserveAndDispatch(Network, CommonMethods):
    
    def __init__(self, n_hours: int, ramping: bool, battery: bool, hydrogen: bool, up_reserve: float, down_reserve: float): # initialize class
        super().__init__()
        
        self.data = expando() # build data attributes
        self.variables = expando() # build variable attributes
        self.constraints = expando() # build sontraint attributes
        self.results = expando()
        self.TIMES = self.TIMES[:n_hours]
        self.ramping = ramping
        self.battery = battery
        self.H2 = hydrogen
        self.up_reserve = up_reserve
        self.down_reserve = down_reserve
        if not battery: 
            self.BATTERIES = []
        self._build_reserve() # build reserve model
        
    
    def _build_reserve(self):
        # initialize optimization model for reserve
        self.model = gb.Model(name='Reserve')
    
        self.variables.generator_up = {(g,t):self.model.addVar(lb=0,ub=self.P_R_PLUS[g],name='Up reserve of generator {0}'.format(g)) for g in self.GENERATORS for t in self.TIMES}
        self.variables.generator_down = {(g,t):self.model.addVar(lb=0,ub=self.P_R_MINUS[g],name='Down reserve of generator {0}'.format(g)) for g in self.GENERATORS for t in self.TIMES}
        
        self.model.update()
        
        # Minimize reserve costs
        self.model.setObjective(gb.quicksum(self.variables.generator_up[g,t] * self.C_U[g] for g in self.GENERATORS for t in self.TIMES)
                                + gb.quicksum(self.variables.generator_down[g,t] * self.C_D[g] for g in self.GENERATORS for t in self.TIMES), gb.GRB.MINIMIZE)
        
        # Compute total demand
        total_demand = {t:gb.quicksum(self.P_D[t][d] for d in self.DEMANDS) for t in self.TIMES}
        
        # Meet up and down reserve requirements
        self.constraints.reserve_up = {(t):self.model.addConstr(gb.quicksum(self.variables.generator_up[g,t] for g in self.GENERATORS), gb.GRB.EQUAL, self.up_reserve * total_demand[t]) for t in self.TIMES}
        
        self.constraints.reserve_down = {(t):self.model.addConstr(gb.quicksum(self.variables.generator_down[g,t] for g in self.GENERATORS), gb.GRB.EQUAL, self.down_reserve * total_demand[t]) for t in self.TIMES}
        
        # Ramping constraints, up reserve
        self.constraints.reserve_up_ramping_dw = {(g,t):self.model.addConstr(
            self.variables.generator_up[g,t] - self.variables.generator_up[g,self.TIMES[n]],
            gb.GRB.GREATER_EQUAL, -self.P_R_DW[g]) for g in self.GENERATORS for n,t in enumerate(self.TIMES[1:])}
        self.constraints.reserve_up_ramping_up = {(g,t):self.model.addConstr(
            self.variables.generator_up[g,t] - self.variables.generator_up[g,self.TIMES[n]],
            gb.GRB.LESS_EQUAL, self.P_R_UP[g]) for g in self.GENERATORS for n,t in enumerate(self.TIMES[1:])}
        
        # Ramping constraints, down reserve
        self.constraints.reserve_down_ramping_dw = {(g,t):self.model.addConstr(
            self.variables.generator_down[g,t] - self.variables.generator_down[g,self.TIMES[n]],
            gb.GRB.GREATER_EQUAL, -self.P_R_DW[g]) for g in self.GENERATORS for n,t in enumerate(self.TIMES[1:])}
        self.constraints.reserve_down_ramping_up = {(g,t):self.model.addConstr(
            self.variables.generator_down[g,t] - self.variables.generator_down[g,self.TIMES[n]],
            gb.GRB.LESS_EQUAL, self.P_R_UP[g]) for g in self.GENERATORS for n,t in enumerate(self.TIMES[1:])}


        # Capacity constraints
        self.constraints.capacity = {(g,t):self.model.addConstr(self.variables.generator_up[g,t] + self.variables.generator_down[g,t], gb.GRB.LESS_EQUAL, self.P_G_max[g]) for g in self.GENERATORS for t in self.TIMES}
        
    def _build_model(self):
        # initialize optimization model for economic dispatch
        self.model = gb.Model(name='Economic Dispatch')
        
        # initialize variables 
        self.variables.consumption = {(d,t):self.model.addVar(lb=0,ub=self.P_D[t][d],name='consumption of demand {0}'.format(d)) for d in self.DEMANDS for t in self.TIMES}
        self.variables.generator_dispatch = {(g,t):self.model.addVar(lb=self.data.down_reserve_values[g,t],ub=(self.P_G_max[g] - self.data.up_reserve_values[g,t]),name='dispatch of generator {0}'.format(g)) for g in self.GENERATORS for t in self.TIMES}
        self.variables.wind_turbines = {(w,t):self.model.addVar(lb=0,ub=self.P_W[t][w],name='dispatch of wind turbine {0}'.format(w)) for w in self.WINDTURBINES for t in self.TIMES}
        if self.H2:
            self.variables.hydrogen = {(w,t):self.model.addVar(lb=0,ub=100,name='consumption of electrolyzer {0}'.format(w)) for w in self.WINDTURBINES for t in self.TIMES}
        if self.battery:
            self.variables.battery_soc = {(b,t):self.model.addVar(lb=0,ub=self.batt_cap[b],name='soc of battery {0}'.format(b)) for b in self.BATTERIES for t in self.TIMES}
            self.variables.battery_ch = {(b,t):self.model.addVar(lb=0,ub=self.batt_power[b],name='dispatch of battery {0}'.format(b)) for b in self.BATTERIES for t in self.TIMES}
            self.variables.battery_dis = {(b,t):self.model.addVar(lb=0,ub=self.batt_power[b],name='consumption of battery {0}'.format(b)) for b in self.BATTERIES for t in self.TIMES}
        
        self.model.update()
        
        # initialize objective to maximize social welfare
        demand_utility = gb.quicksum(self.U_D[t][d] * self.variables.consumption[d,t] for d in self.DEMANDS for t in self.TIMES)
        generator_costs = gb.quicksum(self.C_G_offer[g] * self.variables.generator_dispatch[g,t] for g in self.GENERATORS for t in self.TIMES)
        objective = demand_utility - generator_costs
        self.model.setObjective(objective, gb.GRB.MAXIMIZE)

        # Balance constraint
        # Evaluates based on the values of self.battery and self.H2
        self.constraints.balance_constraint = self.add_balance_constraints()
        
        # ramping constraints
        if self.ramping:
            self.add_ramping_constraints()
        
        # battery constraints
        if self.battery:
            self.add_battery_constraints()
        
        # electrolyzer constraints
        if self.H2:
            self.add_hydrogen_constraints()
        
    def _save_reserve(self):
        # save objective value
        self.data.reserve_cost = self.model.ObjVal
        
        # save up reserve values 
        self.data.up_reserve_values = {(g,t):self.variables.generator_up[g,t].x for g in self.GENERATORS for t in self.TIMES}
        
        # save down reserve values 
        self.data.down_reserve_values = {(g,t):self.variables.generator_down[g,t].x for g in self.GENERATORS for t in self.TIMES}
        
        # save up reserve prices 
        self.data.sigma_up_ = {t:self.constraints.reserve_up[t].Pi for t in self.TIMES}
        
        # save down reserve prices
        self.data.sigma_down_ = {t:self.constraints.reserve_down[t].Pi for t in self.TIMES}

        # save profits of suppliers
        self.data.reserve_profit = {g:sum((self.data.sigma_up_[t] - self.C_U[g]) * self.data.up_reserve_values[g,t] + # Up reserve profit
                                        (self.data.sigma_down_[t] - self.C_D[g]) * self.data.down_reserve_values[g,t] for t in self.TIMES) for g in self.GENERATORS} # Down reserve profit
        
        
    def _save_data(self):
        # save objective value
        self.data.objective_value = self.model.ObjVal
        
        # save consumption values 
        self.data.consumption_values = {(d,t):self.variables.consumption[d,t].x for d in self.DEMANDS for t in self.TIMES}
        
        # save generator dispatches 
        self.data.generator_dispatch_values = {(g,t):self.variables.generator_dispatch[g,t].x for g in self.GENERATORS for t in self.TIMES}
        
        # save wind turbine dispatches 
        self.data.wind_dispatch_values = {(w,t):self.variables.wind_turbines[w,t].x for w in self.WINDTURBINES for t in self.TIMES}
        
        # save battery dispatches 
        if self.battery:
            self.data.battery = {(b,t):self.variables.battery_ch[b,t].x - self.variables.battery_dis[b,t].x for b in self.BATTERIES for t in self.TIMES}
        
        # save electrolyzer activity
        if self.H2:
            self.data.hydrogen = {(w,t):self.variables.hydrogen[w,t].x for w in self.WINDTURBINES for t in self.TIMES}

        
        # save uniform prices lambda 
        self.data.lambda_ = {t:self.constraints.balance_constraint[t].Pi for t in self.TIMES}

        # save profits of suppliers
        self.data.dispatch_profit = {g:sum((self.data.lambda_[t] - self.C_G_offer[g]) * self.data.generator_dispatch_values[g,t] for t in self.TIMES) for g in self.GENERATORS}
        
    
    def run(self):
        self.model.optimize() # Clear reserve market
        self._save_reserve()  # save reserve results
        self._build_model()   # build dispatch model using reserve results
        self.model.optimize() # Clear dispatch market
        self._save_data()     # save dispatch results

    def calculate_results(self):
        # calculate profits of suppliers ( profits = (C_G - lambda) * p_G )
        self.results.profits_G = {g:self.data.reserve_profit[g] + self.data.dispatch_profit[g] for g in self.GENERATORS} # Down reserve profit
        
        self.results.profits_W = {w:sum(self.data.lambda_[t] * self.data.wind_dispatch_values[w,t] for t in self.TIMES) for w in self.WINDTURBINES}
        
        # calculate utility of suppliers ( (U_D - lambda) * p_D )
        self.results.utilities = {d:sum((self.U_D[t][d] - self.data.lambda_[t]) * self.data.consumption_values[d,t] for t in self.TIMES) for d in self.DEMANDS}

    def display_results(self):
        print()
        print("-------------------   RESULTS  -------------------")
        print("Up reserve prices: " + str(self.data.sigma_up_))
        print()
        print("Down reserve prices: " + str(self.data.sigma_down_))
        print()
        print("Market clearing prices: " + str(self.data.lambda_))
        print()
        print("Social welfare: " + str(self.data.objective_value))
        print()
        print("Profit of suppliers: ")
        print("Generators:")
        print(self.results.profits_G)
        print("Wind turbines:")
        print(self.results.profits_W)
        print()
        print("Utility of demands: ")
        print(self.results.utilities)

    def plot_profit(self):
        # Plot grouped bar chart for all generators including wind with profit from reserves, dispatch and total profit.
        # Create a DataFrame for generator profits
        gen_profits = pd.DataFrame([self.results.profits_G, self.data.reserve_profit, self.data.dispatch_profit], index=["Total", "Reserve", "Day-ahead"]).T

        # remove NaN values

        # Create a DataFrame for wind generator profits
        wind_profits = pd.DataFrame([self.results.profits_W], index=["Total"]).T
        wind_profits["Reserve"] = 0
        wind_profits["Day-ahead"] = wind_profits["Total"]

        # Concatenate the two DataFrames
        all_profits = pd.concat([gen_profits, wind_profits])

        # Remove generators with no profit
        all_profits = all_profits.loc[all_profits["Total"] > 0]

        # Plot the profits
        all_profits.plot(kind='bar')
        plt.title("Profits of generators and wind turbines")
        plt.xlabel("Generator / Wind Turbine")
        plt.ylabel("Profit [$]")
        plt.show()

    def plot_prices(self):
        # Plot reserve and market clearing prices
        plt.step(ec.TIMES, list(ec.data.sigma_up_.values()), label='Up Reserve Price [$/MW]')
        plt.step(ec.TIMES, list(ec.data.sigma_down_.values()), label='Down Reserve Price [$/MW]')
        plt.step(ec.TIMES, list(ec.data.lambda_.values()), label='Day-ahead Price [$/MWh]')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        plt.show()
        
        
        
if __name__ == "__main__":
    ec = ReserveAndDispatch(n_hours=24, ramping=True, battery=True, hydrogen=True, up_reserve=0.15, down_reserve=0.1)
    ec.run()
    ec.calculate_results()
    ec.display_results()
    #ec.plot_profit()
    ec.plot_prices()

    


    

    
    

    
    