import gurobipy as gb
from gurobipy import GRB
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from Step_1_2 import Network, expando
from Step_2 import CommonMethods
from network_plots import createNetwork, drawNormal, drawSingleStep, drawLMP

class NodalMarketClearing(Network, CommonMethods):
    
    def __init__(self, model_type: str): # initialize class
        super().__init__()
        
        self.data = expando() # build data attributes
        self.variables = expando() # build variable attributes
        self.constraints = expando() # build sontraint attributes
        self.results = expando()
        self.ramping = True
        self.battery = True
        self.H2 = True
        if not self.battery:
            self.BATTERIES = []
        self.type = model_type
        self._build_model() # build gurobi model
    
    def _build_model(self):
        # initialize optimization model
        self.model = gb.Model(name='Economic Dispatch')
        
        # initialize variables 
        self.variables.consumption = {(d,t):self.model.addVar(lb=0,ub=self.P_D[t][d],name='consumption of demand {0}'.format(d)) for d in self.DEMANDS for t in self.TIMES}
        self.variables.generator_dispatch = {(g,t):self.model.addVar(lb=0,ub=self.P_G_max[g],name='dispatch of generator {0}'.format(g)) for g in self.GENERATORS for t in self.TIMES}
        self.variables.wind_turbines = {(w,t):self.model.addVar(lb=0,ub=self.P_W[t][w],name='dispatch of wind turbine {0}'.format(w)) for w in self.WINDTURBINES for t in self.TIMES}
        if self.type == 'nodal':
            self.variables.theta = {(n,t):self.model.addVar(lb=0,name='voltage angle at node {0}'.format(n)) for n in self.NODES for t in self.TIMES}
        elif self.type == 'zonal':
            self.variables.ic = {(ic,t):self.model.addVar(lb=-self.ic_cap[ic],ub=self.ic_cap[ic],name='interconnector flow {0}'.format(ic)) for ic in self.INTERCONNECTORS for t in self.TIMES}
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
        
        # initialize constraints 
        
        # balance constraint
        if self.type == 'nodal':
            self.constraints.balance_constraint = {(n,t):self.model.addLConstr(
                gb.quicksum(self.variables.consumption[d,t] for d in self.map_d[n])
                - gb.quicksum(self.variables.generator_dispatch[g,t] for g in self.map_g[n])
                - gb.quicksum(self.variables.wind_turbines[w,t] for w in self.map_w[n])
                + gb.quicksum(self.variables.battery_ch[b,t] - self.variables.battery_dis[b,t]
                                  for b in self.map_b[n])
                + gb.quicksum(self.L_susceptance[line]*(self.variables.theta[n,t] - self.variables.theta[m,t]) for m, line in self.map_n[n].items()),
                gb.GRB.EQUAL,
                0, name='Balance equation') for t in self.TIMES for n in self.NODES}
        
        elif self.type == 'zonal':
            self.constraints.balance_constraint = {(z,t):self.model.addLConstr(
                gb.quicksum(self.variables.consumption[d,t] for n in self.map_z[z] for d in self.map_d[n])
                - gb.quicksum(self.variables.generator_dispatch[g,t] for n in self.map_z[z] for g in self.map_g[n])
                - gb.quicksum(self.variables.wind_turbines[w,t] for n in self.map_z[z] for w in self.map_w[n])
                + gb.quicksum(self.variables.battery_ch[b,t] - self.variables.battery_dis[b,t] for n in self.map_z[z] for b in self.map_b[n])
                + gb.quicksum(self.variables.ic[ic,t] for ic in self.zonal[z]) * ((-1) if z == 'Z2' else 1), # direction of ic is towards zone Z2.
                gb.GRB.EQUAL,
                0, name='Balance equation') for t in self.TIMES for z in self.ZONES}
        
        # self.constraints.balance_constraint = {t:self.model.addLConstr(
        #     gb.quicksum(self.variables.consumption[d,t] for d in self.map_d[n])
        #     - gb.quicksum(self.variables.generator_dispatch[g,t] for g in self.map_g[n])
        #     - gb.quicksum(self.variables.wind_turbines[w,t] - self.variables.hydrogen[w,t] for w in self.map_w[n])
        #     + gb.quicksum(self.variables.battery_ch[b,t] - self.variables.battery_dis[b,t]
        #                   for b in self.map_b[n])
        #     + gb.quicksum(self.L_susceptance[line]*(self.variables.theta[n,t] - self.variables.theta[m,t]) for m, line in self.map_n[n].items()),
        #     gb.GRB.EQUAL,
        #     0, name='Balance equation') for t in self.TIMES for n in self.NODES}
        if self.type == 'nodal':
            self.constraints.lines = {(n,m,t): self.model.addLConstr(
                self.L_susceptance[line] * (self.variables.theta[n,t] - self.variables.theta[m,t]),
                gb.GRB.LESS_EQUAL,
                self.L_cap[line],
                name='Line limit') for n in self.NODES for t in self.TIMES for m, line in self.map_n[n].items()}
        
        # ramping constraints
        self.add_ramping_constraints()

        # battery constraints
        self.add_battery_constraints()
        
        # electrolyzer constraints
        self.add_hydrogen_constraints()

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
        if self.type == 'nodal':
            self.data.lambda_ = {t:{n:self.constraints.balance_constraint[n,t].Pi for n in self.NODES} for t in self.TIMES}
        elif self.type == 'zonal':
            self.data.lambda_ = {t:{z:self.constraints.balance_constraint[z,t].Pi for z in self.ZONES} for t in self.TIMES}
        
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
    
    model_type='zonal'
    
    if model_type == 'zonal':
        ec = NodalMarketClearing(model_type)
        ec.run()
        net = createNetwork(ec.map_g, ec.map_d, ec.map_w)
        #drawNormal(net)
        #drawLMP(net, ec.data.lambda_)
        
        # Extract the time steps and nodes
        times = list(ec.data.lambda_.keys())
        zones = list(ec.data.lambda_[times[0]].keys())

        # Define a list of colors and line styles
        colors = ['green', 'blue', 'orange']
        linestyles = ['-', '-', '--']

        # Plot the three zones
        for i, zone in enumerate(zones):
            lambda_values = [ec.data.lambda_[t][zone] for t in times]
            plt.plot(times, lambda_values, drawstyle='steps', label=zone, color=colors[i], linestyle=linestyles[i], linewidth=3)

        # Add labels and legend
        plt.ylabel('Price [$/MWh]')
        plt.xlabel('Time')
        plt.legend() 
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.1)       
        plt.show()
        
    elif model_type == 'nodal':
    
        ec = NodalMarketClearing(model_type)
        ec.run()
        # net = createNetwork(ec.map_g, ec.map_d, ec.map_w)
        # drawNormal(net)
        # drawLMP(net, ec.data.lambda_)
        
        
        # Extract the time steps and nodes
        times = list(ec.data.lambda_.keys())
        nodes = list(ec.data.lambda_[times[0]].keys())

        # Define a list of colors and line styles
        colors = ['blue', 'green', 'red', 'orange', 'purple', 'cyan']
        linestyles = ['-', '--']

        # Plot the nodal time series as stairs plots with unique colors and line styles
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

        # Plot the first 12 graphs in the upper plot
        for i, node in enumerate(nodes[:12]):
            lambda_values = [ec.data.lambda_[t][node] for t in times]
            color = colors[i % len(colors)]
            linestyle = linestyles[i//6 % len(linestyles)]
            ax1.plot(times, lambda_values, drawstyle='steps', label=node, color=color, linestyle=linestyle)

        # Plot the remaining graphs in the lower plot
        for i, node in enumerate(nodes[12:]):
            lambda_values = [ec.data.lambda_[t][node] for t in times]
            color = colors[i % len(colors)]
            linestyle = linestyles[i//6 % len(linestyles)]
            ax2.plot(times, lambda_values, drawstyle='steps', label=node, color=color, linestyle=linestyle)

        # Add labels and legend
        ax1.set_ylabel('Price [$/MWh]', fontsize=16)
        ax2.set_xlabel('Time', fontsize=16)
        ax2.set_ylabel('Price [$/MWh]', fontsize=16)
        ax1.legend(loc = 'upper left', fontsize=12)
        ax2.legend(loc = 'upper left', fontsize=12)

        # Show the plot
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.1)
        plt.show()
