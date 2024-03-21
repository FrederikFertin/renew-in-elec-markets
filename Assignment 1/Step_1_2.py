import gurobipy as gb
from gurobipy import GRB
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Step_2 import CommonMethods
from network_plots import plot_SD_curve
import os

class Network:
    # Reading data from Excel, requires openpyxl
    cwd = os.getcwd()
    xls = pd.ExcelFile(cwd + '/data.xlsx')
    #xls = pd.ExcelFile('data.xlsx')
    gen_tech = pd.read_excel(xls, 'gen_technical')
    gen_econ = pd.read_excel(xls, 'gen_cost')
    system_demand = pd.read_excel(xls, 'demand')
    line_info = pd.read_excel(xls, 'transmission_lines')
    load_info = pd.read_excel(xls, 'demand_nodes')
    wind_tech = pd.read_excel(xls, 'wind_technical')

    # Loading csv file of normalized wind profiles
    wind_profiles = pd.read_csv(cwd + '/wind_profiles.csv')
    #wind_profiles = pd.read_csv('wind_profiles.csv')

    # Number of each type of unit/identity
    G = np.shape(gen_tech)[0] # Number of generators
    D = np.shape(load_info)[0] # Number of loads/demands
    T = np.shape(system_demand)[0] # Number of time periods/hours
    L = np.shape(line_info)[0] # Number of transmission lines
    W = np.shape(wind_tech)[0] # Number of wind farms
    N = 24 # Number of nodes in network

    # Lists of Generators etc.
    GENERATORS = ['G{0}'.format(t) for t in range(1, G+1)]
    DEMANDS = ['D{0}'.format(t) for t in range(1, D+1)]
    LINES = ['L{0}'.format(t) for t in range(1, L+1)]
    TIMES = ['T{0}'.format(t) for t in range(1, T+1)]
    WINDTURBINES = ['W{0}'.format(t) for t in range(1, W+1)]
    NODES = ['N{0}'.format(t) for t in range(1, N+1)]
    ZONES = ['Z1', 'Z2', 'Z3']
    
    map_z = {'Z1': ['N17', 'N18', 'N21', 'N22'],
             'Z2': ['N11', 'N12', 'N13', 'N14', 'N15', 'N16', 'N19', 'N20', 'N23', 'N24'],
             'Z3': ['N{0}'.format(t) for t in range(1, 11)]}

    map_nz = {n: z for z, ns in map_z.items() for n in ns}
    
    ## Conventional Generator Information
    P_G_max = dict(zip(GENERATORS, gen_tech['P_max'])) # Max generation cap.
    P_G_min = dict(zip(GENERATORS, gen_tech['P_min'])) # Min generation cap.
    C_G_offer = dict(zip(GENERATORS, gen_econ['C'])) # Generator day-ahead offer price
    P_R_DW = dict(zip(GENERATORS, gen_tech['R_D'])) # Up-ramping of generator
    P_R_UP = dict(zip(GENERATORS, gen_tech['R_U'])) # Down-ramping of generator
    node_G = dict(zip(GENERATORS, gen_tech['Node'])) # Generator node placements
    P_R_PLUS = dict(zip(GENERATORS, gen_tech['R_plus'])) # Up reserve capacity
    P_R_MINUS = dict(zip(GENERATORS, gen_tech['R_minus'])) # Down reserve capacity
    C_U = dict(zip(GENERATORS, gen_econ['C_u'])) # Up reserve cost
    C_D = dict(zip(GENERATORS, gen_econ['C_d'])) # Down reserve cost

    
    ## Demand Information
    P_D_sum = dict(zip(TIMES, system_demand['System_demand'])) # Total system demands
    P_D = {} # Distribution of system demands
    for t, key in enumerate(TIMES):
        P_D[key] = dict(zip(DEMANDS, load_info['load_percent']/100*system_demand['System_demand'][t]))
    U_D = {}
    for t, key in enumerate(TIMES):
        U_D[key] = dict(zip(DEMANDS, load_info['bid_price'])) # Demand bidding price <- set values in excel
    U_D['T9']['D13'] = 10.2
    U_D['T9']['D16'] = 7.0
    node_D = dict(zip(DEMANDS, load_info['Node'])) # Load node placements
    U_D_curt = 400 # cost of demand curtailment in BM [$/MWh]
    
    ## Wind Turbine Information
    p_W_cap = 200 # Wind farm capacities (MW)
    WT = ['V{0}'.format(v) for v in wind_tech['Profile']]
    chosen_wind_profiles = wind_profiles[WT] # 'Randomly' chosen profiles for each wind farm
    P_W = {} # Wind production for each hour and each wind farm
    for t, key in enumerate(TIMES):
        P_W[key] = dict(zip(WINDTURBINES, chosen_wind_profiles.iloc[t,:] * p_W_cap))
    node_W = dict(zip(WINDTURBINES, wind_tech['Node'])) # Wind turbine node placements
    
    ## Electrolyzer Information
    hydrogen_daily_demand = 100*0.2*24 # 8160 kg of hydrogen
    
    ## Battery Information
    BATTERIES = ['B1']
    batt_cap = {'B1': 400} # Battery capacity is 400 MWh
    batt_init_soc = {'B1': 200} # Initial state of charge of battery - at time t-1 (T0)
    batt_power = {'B1': 200} # Battery (dis)charging limit is 200 MW
    batt_node = {'B1': 11} # Battery is placed at node 11
    batt_eta = {'B1': 0.99} # Battery charging and discharging efficiency of 95%

    ## Transmission Line Information
    L_cap = dict(zip(LINES, line_info['Capacity_wind'])) # Capacity of transmission line [MVA]
    # R_base = 
    L_susceptance = dict(zip(LINES, [500 for i in LINES])) # 1/line_info['Reactance'])) #  Susceptance of transmission line [pu.] 
    L_from = dict(zip(LINES, line_info['From'])) # Origin node of transmission line
    L_to = dict(zip(LINES, line_info['To'])) # Destination node of transmission line
    
    ## Inter-Zonal capacitances
    c_z1_z2 = L_cap['L25'] + L_cap['L27']
    c_z2_z3 = L_cap['L7'] + L_cap['L14'] + L_cap['L15'] + L_cap['L16'] + L_cap['L17']
    ZONES = ['Z1', 'Z2', 'Z3']
    zone_cap = {'Z1': {'Z2': c_z1_z2},
                  'Z2': {'Z1': c_z1_z2, 'Z3': c_z2_z3},
                  'Z3': {'Z2': c_z2_z3}}
    zonal = {'Z1': ['Z12'],
             'Z2': ['Z12', 'Z23'],
             'Z3': ['Z23']}
    INTERCONNECTORS = ['Z12', 'Z23']
    ic_cap = {'Z12': c_z1_z2,
              'Z23': c_z2_z3}


    def __init__(self):
        # Nodal mappings:
        self.map_g = self._map_units(self.node_G)
        self.map_d = self._map_units(self.node_D)
        self.map_w = self._map_units(self.node_W)
        self.map_b = self._map_units(self.batt_node)
        self.map_from = self._map_units(self.L_from)
        self.map_to = self._map_units(self.L_to)
        self._map_nodes()

    def _map_units(self,node_list):
        mapping_units = {}
        for number, node in enumerate(self.NODES):
            n = number + 1
            u_list = []
            for k, v in node_list.items():
                if v == n:
                    u_list.append(k)
            mapping_units[node] = u_list
        return mapping_units
    
    def _map_nodes(self):
        self.map_n = {}
        for node_to, lines in self.map_to.items():
            self.map_n[node_to] = {}
            for line in lines:
                for node_from, lines_from in self.map_from.items():
                    if line in lines_from:
                        self.map_n[node_to][node_from] = line
        for node_from, lines in self.map_from.items():
            for line in lines:
                for node_to, lines_to in self.map_to.items():
                    if line in lines_to:
                        self.map_n[node_from][node_to] = line


class expando(object):
    '''
        A small class which can have attributes set
    '''
    pass

class EconomicDispatch(Network, CommonMethods):
    
    def __init__(self, n_hours: int, ramping: bool, battery: bool, hydrogen: bool): # initialize class
        super().__init__()
        
        self.data = expando() # build data attributes
        self.variables = expando() # build variable attributes
        self.constraints = expando() # build sontraint attributes
        self.results = expando()
        self.TIMES = self.TIMES[:n_hours]
        self.ramping = ramping
        self.battery = battery
        self.H2 = hydrogen
        if not battery: 
            self.BATTERIES = []
        self._build_model() # build gurobi model
    
    def _build_model(self):
        # initialize optimization modenb  bbn l
        self.model = gb.Model(name='Economic Dispatch')
        
        # initialize variables 
        self.variables.consumption = {(d,t):self.model.addVar(lb=0,ub=self.P_D[t][d],name='consumption of demand {0}'.format(d)) for d in self.DEMANDS for t in self.TIMES}
        self.variables.generator_dispatch = {(g,t):self.model.addVar(lb=0,ub=self.P_G_max[g],name='dispatch of generator {0}'.format(g)) for g in self.GENERATORS for t in self.TIMES}
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
        
        # initialize constraints
        # initialize constraints 
        ## Step 1 - getting duals for the marginal generator:
        #self.constraints.generation_constraint_min = {t:self.model.addConstr(
        #    self.variables.generator_dispatch['G7',t], gb.GRB.GREATER_EQUAL, 0.1, name='Min gen G7') for t in self.TIMES}
        #self.constraints.generation_constraint_max = {t:self.model.addConstr(
        #    self.variables.generator_dispatch['G7',t], gb.GRB.LESS_EQUAL, self.P_G_max['G7']-0.1, name='Max gen G7') for t in self.TIMES}
        
        # Balance constraints
        # Evaluates based on the values of self.battery and self.H2
        self.constraints.balance_constraint = self._add_balance_constraints()
        
        # ramping constraints
        if self.ramping:
            self._add_ramping_constraints()

        # battery constraints
        if self.battery:
            self._add_battery_constraints()
        
        # electrolyzer constraints
        if self.H2:
            self._add_hydrogen_constraints()
        
    def _save_data(self):
        # save objective value
        self.data.objective_value = self.model.ObjVal
        
        # save consumption values 
        self.data.consumption_values = {(d,t):self.variables.consumption[d,t].x for d in self.DEMANDS for t in self.TIMES}
        
        # save generator dispatches 
        self.data.generator_dispatch_values = {(g,t):self.variables.generator_dispatch[g,t].x for g in self.GENERATORS for t in self.TIMES}
        
        # save wind turbine dispatches 
        self.data.wind_dispatch_values = {(w,t):self.variables.wind_turbines[w,t].x for w in self.WINDTURBINES for t in self.TIMES}

        # save up and down regulation constraints        
        if self.ramping:
            self.data.ramping_up_dual = {t : {g:self.constraints.ramping_up[g,t].Pi for g in self.GENERATORS} for t in self.TIMES[1:]}
            self.data.ramping_dw_dual = {t : {g:self.constraints.ramping_dw[g,t].Pi for g in self.GENERATORS} for t in self.TIMES[1:]}

        # save battery dispatches
        if self.battery:
            self.data.battery = {(b,t):self.variables.battery_ch[b,t].x - self.variables.battery_dis[b,t].x for b in self.BATTERIES for t in self.TIMES}
            self.data.battery_soc = {(b,t):self.variables.battery_soc[b,t].x for b in self.BATTERIES for t in self.TIMES}
            self.data.battery_soc_constraint = {(b,t):self.constraints.batt_soc[b,t].Pi for t in self.TIMES[1:] for b in self.BATTERIES}
        
        # save electrolyzer activity
        if self.H2:
            self.data.hydrogen = {(w,t):self.variables.hydrogen[w,t].x for w in self.WINDTURBINES for t in self.TIMES}

        
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
        self.results.utilities = {d:sum((self.U_D[t][d] - self.data.lambda_[t]) * self.data.consumption_values[d,t] for t in self.TIMES) for d in self.DEMANDS}

    def display_results(self):
        print()
        print("-------------------   RESULTS  -------------------")
        print("Market clearing prices: " + str(self.data.lambda_))
        print()
        print("Social welfare: " + str(self.data.objective_value))
        print()
        print("Profit of suppliers: ")
        print("Generators: ")
        print(self.results.profits_G)
        print("Wind turbines: ")
        print(self.results.profits_W)
        print()
        print("Utility of demands: ")
        print(self.results.utilities)

if __name__ == "__main__":
    #ec = EconomicDispatch(n_hours=1, ramping=False, battery=False, hydrogen=False)

    # ec.run()
    # ec.calculate_results()
    # ec.display_results()
    ec = EconomicDispatch(n_hours=24, ramping=True, battery=True, hydrogen=True)
    ec.run()
    ec.calculate_results()
    ec.display_results()
    
    #print(ec.data.battery)
    #plot_SD_curve(ec, 'T16')




