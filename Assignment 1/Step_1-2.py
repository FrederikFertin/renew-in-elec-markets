import gurobipy as gb
from gurobipy import GRB
import numpy as np
import pandas as pd

class Network:
    # Reading data from Excel, requires openpyxl
    
    # xls = pd.ExcelFile('Assignment 1/data.xlsx')
    xls = pd.ExcelFile('data.xlsx')
    gen_tech = pd.read_excel(xls, 'gen_technical')
    gen_econ = pd.read_excel(xls, 'gen_cost')
    system_demand = pd.read_excel(xls, 'demand')
    line_info = pd.read_excel(xls, 'transmission_lines')
    load_info = pd.read_excel(xls, 'demand_nodes')
    wind_tech = pd.read_excel(xls, 'wind_technical')

    # Loading csv file of normalized wind profiles
    # wind_profiles = pd.read_csv('Assignment 1/wind_profiles.csv')
    wind_profiles = pd.read_csv('wind_profiles.csv')

    # Number of each type of unit/identity
    G = np.shape(gen_tech)[0] # Number of generators
    D = np.shape(load_info)[0] # Number of loads/demands
    T = np.shape(system_demand)[0] # Number of time periods/hours
    L = np.shape(line_info)[0] # Number of transmission lines
    W = np.shape(wind_tech)[0] # Number of wind farms
    # N = 24 # Number of nodes in network

    # Lists of Generators etc.
    GENERATORS = ['G{0}'.format(t) for t in range(1, G+1)]
    DEMANDS = ['D{0}'.format(t) for t in range(1, D+1)]
    LINES = ['L{0}'.format(t) for t in range(1, L+1)]
    TIMES = ['T{0}'.format(t) for t in range(1, T+1)]
    WINDTURBINES = ['W{0}'.format(t) for t in range(1, W+1)]
    # NODES = ['N{0}'.format(t) for t in range(1, N)]

    ## Conventional Generator Information
    P_G_max = dict(zip(GENERATORS, gen_tech['P_max'])) # Max generation cap.
    C_G_offer = dict(zip(GENERATORS, gen_econ['C'])) # Generator day-ahead offer price
    
    ## Demand Information
    P_D_sum = dict(zip(TIMES, system_demand['System_demand'])) # Total system demands
    P_D = {} # Distribution of system demands
    for t, key in enumerate(TIMES):
        P_D[key] = dict(zip(DEMANDS, load_info['load_percent']/100*system_demand['System_demand'][t]))
    U_D = dict(zip(DEMANDS, load_info['bid_price'])) # Demand bidding price <- set values in excel

    ## Wind Turbine Information
    p_W_cap = 200 # Wind farm capacities (MW)
    WT = ['V{0}'.format(v) for v in wind_tech['Profile']]
    chosen_wind_profiles = wind_profiles[WT] # 'Randomly' chosen profiles for each wind farm
    P_W = {} # Wind production for each hour and each wind farm
    for t, key in enumerate(TIMES):
        P_W[key] = dict(zip(WINDTURBINES, chosen_wind_profiles.iloc[t,:] * p_W_cap))


    """
    # Fraction of system consumption at each node indexed by loads 
    P_D_fraction = np.array([0.038, 0.034, 0.063, 0.026, 0.025, 0.048, 0.044, 0.06,
                             0.061, 0.068, 0.093, 0.068, 0.111, 0.035, 0.117, 0.064,
                             0.045])

    # Max generation indexed by generator
    P_G_max = {'G1': 152, 'G2': 152, 'G3': 350, 'G4': 591, 'G5': 60, 'G6': 155,
               'G7': 155, 'G8': 400, 'G9': 400, 'G10': 300, 'G11': 310, 'G12': 350}

    # Demand quantities indexed by demand
    P_D = {'D1': P_D_sum}

    P_R_DW = {k: -v / 2 for k, v in P_G_max.items()}
    P_R_UP = {k: v / 2 for k, v in P_G_max.items()}
    """

class expando(object):
    '''
        A small class which can have attributes set
    '''
    pass

class EconomicDispatch(Network):
    
    def __init__(self, n_hours: int, ramping: bool, battery: bool): # initialize class
        # super().__init__(n_samples=n_samples)
        
        self.data = expando() # build data attributes
        self.variables = expando() # build variable attributes
        self.constraints = expando() # build sontraint attributes
        self.results = expando()
        self.TIMES = self.TIMES[:n_hours]
        self.ramping = ramping
        self.battery = battery
        self._build_model() # build gurobi model
    
    def _build_model(self):
        # initialize optimization model
        self.model = gb.Model(name='Economic Dispatch')
        
        # initialize variables 
        self.variables.consumption = {(d,t):self.model.addVar(lb=0,ub=self.P_D[t][d],name='consumption of demand {0}'.format(d)) for d in self.DEMANDS for t in self.TIMES}
        self.variables.generator_dispatch = {(g,t):self.model.addVar(lb=0,ub=self.P_G_max[g],name='dispatch of generator {0}'.format(g)) for g in self.GENERATORS for t in self.TIMES}
        self.variables.wind_turbines = {(w,t):self.model.addVar(lb=0,ub=self.P_W[t][w],name='dispatch of wind turbine {0}'.format(w)) for w in self.WINDTURBINES for t in self.TIMES}
        if self.battery:
            self.variables.battery_soc = {(b,t):self.model.addVar(lb=0,ub=self.batt_cap[b],name='soc of battery {0}'.format(b)) for b in self.BATTERIES for t in self.TIMES}
            self.variables.battery_ch = {(b,t):self.model.addVar(lb=0,ub=self.batt_power[b],name='dispatch of battery {0}'.format(b)) for b in self.BATTERIES for t in self.TIMES}
            self.variables.battery_dis = {(b,t):self.model.addVar(lb=0,ub=self.batt_power[b],name='consumption of battery {0}'.format(b)) for b in self.BATTERIES for t in self.TIMES}
        
        # initialize objective to maximize social welfare
        demand_utility = gb.quicksum(self.U_D[d] * self.variables.consumption[d,t] for d in self.DEMANDS for t in self.TIMES)
        generator_costs = gb.quicksum(self.C_G_offer[g] * self.variables.generator_dispatch[g,t] for g in self.GENERATORS for t in self.TIMES)
        objective = demand_utility - generator_costs
        self.model.setObjective(objective, gb.GRB.MAXIMIZE)
        
        # initialize constraints 
        self.constraints.balance_constraint = {t:self.model.addLConstr(
                gb.quicksum(self.variables.consumption[d,t] for d in self.DEMANDS)
                - gb.quicksum(self.variables.generator_dispatch[g,t] for g in self.GENERATORS)
                - gb.quicksum(self.variables.wind_turbines[w,t] for w in self.WINDTURBINES),
                gb.GRB.EQUAL,
                0, name='Balance equation') for t in self.TIMES}

        T = self.TIMES
        if self.ramping:
            self.constraints.ramping_dw = {(g,t):self.model.addConstr(
                self.variables.generator_dispatch[g,t] - self.variables.generator_dispatch[g,T[n]],
                gb.GRB.GREATER_EQUAL,
                self.P_R_DW[g]) for g in self.GENERATORS for n,t in enumerate(self.TIMES[1:])}
            self.constraints.ramping_up = {(g,t):self.model.addConstr(
                self.variables.generator_dispatch[g,t] - self.variables.generator_dispatch[g,T[n]],
                gb.GRB.LESS_EQUAL,
                self.P_R_UP[g]) for g in self.GENERATORS for n,t in enumerate(self.TIMES[1:])}
        if self.battery:
            self.constraints.batt_soc = {(b,t):self.model.addConstr(
                self.variables.battery_soc[b,t], 
                gb.GRB.EQUAL, 
                self.variables.battery_soc[b,t-1] + batt_eta[b] * self.variables.battery_ch[b,t] - 1/batt_eta[b] * self.variables.battery_dis[b,t])
                for b in self.BATTERIES for t in self.TIMES[1:]}
            self.constraints.init_batt_soc = {(b):self.model.addConstr(
                self.variables.battery_soc[b,0], 
                gb.GRB.EQUAL, 
                self.batt_init_soc[b] + batt_eta[b] * self.variables.battery_ch[b,0] - 1/batt_eta[b] * self.variables.battery_dis[b,0])
                for b in self.BATTERIES}
            self.constraints.final_batt_soc = {(b):self.model.addConstr(
                self.variables.battery_soc[b,self.TIMES[-1]],
                gb.GRB.GREATER_EQUAL,
                self.batt_init_soc[b])
                for b in self.BATTERIES}
        
    def _save_data(self):
        # save objective value
        self.data.objective_value = self.model.ObjVal
        
        # save consumption values 
        self.data.consumption_values = {(d,t):self.variables.consumption[d,t].x for d in self.DEMANDS for t in self.TIMES}
        
        # save generator dispatches 
        self.data.generator_dispatch_values = {(g,t):self.variables.generator_dispatch[g,t].x for g in self.GENERATORS for t in self.TIMES}
        
        # save wind turbine dispatches 
        self.data.wind_dispatch_values = {(w,t):self.variables.wind_turbines[w,t].x for w in self.WINDTURBINES for t in self.TIMES}
        
        # save uniform prices lambda 
        self.data.lambda_ = {t:self.constraints.balance_constraint[t].Pi for t in self.TIMES}
        
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
    ec = EconomicDispatch(n_hours=24, ramping=False)
    ec.run()
    ec.calculate_results()
    ec.display_results()













































        
