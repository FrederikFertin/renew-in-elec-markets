import gurobipy as gb
from gurobipy import GRB
import numpy as np
import pandas as pd
from Step_1_2 import Network, expando

class NodalMarketClearing(Network):
    
    def __init__(self, ramping: bool, battery: bool, hydrogen: bool): # initialize class
        super().__init__()
        
        self.data = expando() # build data attributes
        self.variables = expando() # build variable attributes
        self.constraints = expando() # build sontraint attributes
        self.results = expando()
        self.ramping = ramping
        self.battery = battery
        self.H2 = hydrogen
        if not battery:
            self.BATTERIES = []
        self._build_model() # build gurobi model
    
    def _build_model(self):
        # initialize optimization model
        self.model = gb.Model(name='Economic Dispatch')
        
        # initialize variables 
        self.variables.consumption = {(d,t):self.model.addVar(lb=0,ub=self.P_D[t][d],name='consumption of demand {0}'.format(d)) for d in self.DEMANDS for t in self.TIMES}
        self.variables.generator_dispatch = {(g,t):self.model.addVar(lb=0,ub=self.P_G_max[g],name='dispatch of generator {0}'.format(g)) for g in self.GENERATORS for t in self.TIMES}
        self.variables.wind_turbines = {(w,t):self.model.addVar(lb=0,ub=self.P_W[t][w],name='dispatch of wind turbine {0}'.format(w)) for w in self.WINDTURBINES for t in self.TIMES}
        self.variables.theta = {(n,t):self.model.addVar(lb=0,name='voltage angle at node {0}'.format(n)) for n in self.NODES for t in self.TIMES}
        if self.H2:
            self.variables.hydrogen = {(w,t):self.model.addVar(lb=0,ub=100,name='consumption of electrolyzer {0}'.format(w)) for w in self.WINDTURBINES for t in self.TIMES}
        if self.battery:
            self.variables.battery_soc = {(b,t):self.model.addVar(lb=0,ub=self.batt_cap[b],name='soc of battery {0}'.format(b)) for b in self.BATTERIES for t in self.TIMES}
            self.variables.battery_ch = {(b,t):self.model.addVar(lb=0,ub=self.batt_power[b],name='dispatch of battery {0}'.format(b)) for b in self.BATTERIES for t in self.TIMES}
            self.variables.battery_dis = {(b,t):self.model.addVar(lb=0,ub=self.batt_power[b],name='consumption of battery {0}'.format(b)) for b in self.BATTERIES for t in self.TIMES}
        
        self.model.update()
        
        # initialize objective to maximize social welfare
        demand_utility = gb.quicksum(self.U_D[d] * self.variables.consumption[d,t] for d in self.DEMANDS for t in self.TIMES)
        generator_costs = gb.quicksum(self.C_G_offer[g] * self.variables.generator_dispatch[g,t] for g in self.GENERATORS for t in self.TIMES)
        objective = demand_utility - generator_costs
        self.model.setObjective(objective, gb.GRB.MAXIMIZE)
        
        # initialize constraints 
        
        # balance constraint
        self.constraints.balance_constraint = {(n,t):self.model.addLConstr(
            gb.quicksum(self.variables.consumption[d,t] for d in self.map_d[n])
            - gb.quicksum(self.variables.generator_dispatch[g,t] for g in self.map_g[n])
            - gb.quicksum(self.variables.wind_turbines[w,t] for w in self.map_w[n])
            + gb.quicksum(self.variables.battery_ch[b,t] - self.variables.battery_dis[b,t]
                              for b in self.map_b[n])
            + gb.quicksum(self.L_susceptance[line]*(self.variables.theta[n,t] - self.variables.theta[m,t]) for m, line in self.map_n[n].items()),
            gb.GRB.EQUAL,
            0, name='Balance equation') for t in self.TIMES for n in self.NODES}

        # self.constraints.balance_constraint = {t:self.model.addLConstr(
        #     gb.quicksum(self.variables.consumption[d,t] for d in self.map_d[n])
        #     - gb.quicksum(self.variables.generator_dispatch[g,t] for g in self.map_g[n])
        #     - gb.quicksum(self.variables.wind_turbines[w,t] - self.variables.hydrogen[w,t] for w in self.map_w[n])
        #     + gb.quicksum(self.variables.battery_ch[b,t] - self.variables.battery_dis[b,t]
        #                   for b in self.map_b[n])
        #     + gb.quicksum(self.L_susceptance[line]*(self.variables.theta[n,t] - self.variables.theta[m,t]) for m, line in self.map_n[n].items()),
        #     gb.GRB.EQUAL,
        #     0, name='Balance equation') for t in self.TIMES for n in self.NODES}

        self.constraints.lines = {(n,m,t): self.model.addLConstr(
            self.L_susceptance[line] * (self.variables.theta[n,t] - self.variables.theta[m,t]),
            gb.GRB.LESS_EQUAL,
            self.L_cap[line],
            name='Line limit') for n in self.NODES for t in self.TIMES for m, line in self.map_n[n].items()}
        
        # ramping constraints
        T = self.TIMES
        if self.ramping:
            self.constraints.ramping_dw = {(g,t):self.model.addConstr(
                self.variables.generator_dispatch[g,t] - self.variables.generator_dispatch[g,T[n]],
                gb.GRB.GREATER_EQUAL,
                -self.P_R_DW[g]) for g in self.GENERATORS for n,t in enumerate(self.TIMES[1:])}
            self.constraints.ramping_up = {(g,t):self.model.addConstr(
                self.variables.generator_dispatch[g,t] - self.variables.generator_dispatch[g,T[n]],
                gb.GRB.LESS_EQUAL,
                self.P_R_UP[g]) for g in self.GENERATORS for n,t in enumerate(self.TIMES[1:])}
            
        # battery constraints
        if self.battery:
            # soc constraint
            self.constraints.batt_soc = {(b,t):self.model.addLConstr(
                self.variables.battery_soc[b,t], 
                gb.GRB.EQUAL,
                self.variables.battery_soc[b,T[n]] + self.batt_eta[b] * self.variables.battery_ch[b,t] - 1/self.batt_eta[b] * self.variables.battery_dis[b,t])
                for b in self.BATTERIES for n,t in enumerate(self.TIMES[1:])}
            # initializing soc constraint
            self.constraints.init_batt_soc = {(b):self.model.addLConstr(
                self.variables.battery_soc[b,self.TIMES[0]], 
                gb.GRB.EQUAL, 
                self.batt_init_soc[b] + self.batt_eta[b] * self.variables.battery_ch[b,self.TIMES[0]] - 1/self.batt_eta[b] * self.variables.battery_dis[b,self.TIMES[0]])
                for b in self.BATTERIES}
            # final soc constraint
            self.constraints.final_batt_soc = {(b):self.model.addLConstr(
                self.variables.battery_soc[b,self.TIMES[-1]],
                gb.GRB.GREATER_EQUAL,
                self.batt_init_soc[b])
                for b in self.BATTERIES}
        
        # electrolyzer constraints
        if self.H2:
            self.constraints.hydrogen_limit = {(w,t):self.model.addLConstr(
                self.variables.hydrogen[w,t],
                gb.GRB.LESS_EQUAL,
                self.P_W[t][w])
                for t in self.TIMES for w in self.WINDTURBINES}
            self.constraints.hydrogen_sum = {(w):self.model.addLConstr(
                gb.quicksum(self.variables.hydrogen[w,t] for t in self.TIMES),
                gb.GRB.GREATER_EQUAL,
                self.hydrogen_daily_demand) 
                for w in self.WINDTURBINES}
        
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
        self.data.lambda_ = {(n,t):self.constraints.balance_constraint[n,t].Pi for n in self.NODES for t in self.TIMES}
        
    def run(self):
        self.model.optimize()
        self._save_data()

    def calculate_results(self):
        # calculate profits of suppliers ( profits = (C_G - lambda) * p_G )
        self.results.profits_G = {g:sum((self.data.lambda_[t] - self.C_G_offer[g])  * self.data.generator_dispatch_values[g,t] for t in self.TIMES) for g in self.GENERATORS}
        self.results.profits_W = {w:sum(self.data.lambda_[t] * self.data.wind_dispatch_values[w,t] for t in self.TIMES) for w in self.WINDTURBINES}
        
        # calculate utility of suppliers ( (U_D - lambda) * p_D )
        self.results.utilities = {d:sum((self.U_D[d] - self.data.lambda_[t]) * self.data.consumption_values[d,t] for t in self.TIMES) for d in self.DEMANDS}

    def display_results(self):
        print()
        print("-------------------   RESULTS  -------------------")
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
        
if __name__ == "__main__":
    ec = NodalMarketClearing(ramping=True, battery=True, hydrogen=True)
    ec.run()


