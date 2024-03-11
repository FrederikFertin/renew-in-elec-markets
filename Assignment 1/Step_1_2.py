import gurobipy as gb
from gurobipy import GRB
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Network:
    # Reading data from Excel, requires openpyxl
    
    xls = pd.ExcelFile('Assignment 1/data.xlsx')
    # xls = pd.ExcelFile('data.xlsx')
    gen_tech = pd.read_excel(xls, 'gen_technical')
    gen_econ = pd.read_excel(xls, 'gen_cost')
    system_demand = pd.read_excel(xls, 'demand')
    line_info = pd.read_excel(xls, 'transmission_lines')
    load_info = pd.read_excel(xls, 'demand_nodes')
    wind_tech = pd.read_excel(xls, 'wind_technical')

    # Loading csv file of normalized wind profiles
    wind_profiles = pd.read_csv('Assignment 1/wind_profiles.csv')
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
    
    ## Conventional Generator Information
    P_G_max = dict(zip(GENERATORS, gen_tech['P_max'])) # Max generation cap.
    C_G_offer = dict(zip(GENERATORS, gen_econ['C'])) # Generator day-ahead offer price
    P_R_DW = dict(zip(GENERATORS, gen_tech['R_D'])) # Up-ramping of generator
    P_R_UP = dict(zip(GENERATORS, gen_tech['R_U'])) # Down-ramping of generator
    node_G = dict(zip(GENERATORS, gen_tech['Node'])) # Generator node placements
    
    
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
    batt_eta = {'B1': 0.95} # Battery charging and discharging efficiency of 95%
    batt_eta = {'B1': 0.99} # Battery charging and discharging efficiency of 95%

    ## Transmission Line Information
    L_cap = dict(zip(LINES, line_info['Capacity_wind'])) # Capacity of transmission line [MVA]
    # R_base = 
    L_susceptance = dict(zip(LINES, 1/line_info['Reactance'])) # [500 for i in LINES])) Susceptance of transmission line [pu.]
    L_from = dict(zip(LINES, line_info['From'])) # Origin node of transmission line
    L_to = dict(zip(LINES, line_info['To'])) # Destination node of transmission line
    
    ## Inter-Zonal capacitances
    c_z1_z2 = L_cap['L25'] + L_cap['L27']
    c_z2_z3 = 2000
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
        # initialize optimization model
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
        
        # balance constraint
        if self.battery and self.H2:
            self.constraints.balance_constraint = {t:self.model.addLConstr(
                    gb.quicksum(self.variables.consumption[d,t] for d in self.DEMANDS)
                    - gb.quicksum(self.variables.generator_dispatch[g,t] for g in self.GENERATORS)
                    - gb.quicksum(self.variables.wind_turbines[w,t] - self.variables.hydrogen[w,t] for w in self.WINDTURBINES)
                    + gb.quicksum(self.variables.battery_ch[b,t] - self.variables.battery_dis[b,t] 
                                  for b in self.BATTERIES),
                    gb.GRB.EQUAL,
                    0, name='Balance equation') for t in self.TIMES}
        elif self.battery:
            self.constraints.balance_constraint = {t:self.model.addLConstr(
                    gb.quicksum(self.variables.consumption[d,t] for d in self.DEMANDS)
                    - gb.quicksum(self.variables.generator_dispatch[g,t] for g in self.GENERATORS)
                    - gb.quicksum(self.variables.wind_turbines[w,t] for w in self.WINDTURBINES)
                    + gb.quicksum(self.variables.battery_ch[b,t] - self.variables.battery_dis[b,t] 
                                  for b in self.BATTERIES),
                    gb.GRB.EQUAL,
                    0, name='Balance equation') for t in self.TIMES}
        elif self.H2:
            self.constraints.balance_constraint = {t:self.model.addLConstr(
                    gb.quicksum(self.variables.consumption[d,t] for d in self.DEMANDS)
                    - gb.quicksum(self.variables.generator_dispatch[g,t] for g in self.GENERATORS)
                    - gb.quicksum(self.variables.wind_turbines[w,t] - self.variables.hydrogen[w,t] for w in self.WINDTURBINES),
                    gb.GRB.EQUAL,
                    0, name='Balance equation') for t in self.TIMES}
        else:
            self.constraints.balance_constraint = {t:self.model.addLConstr(
                    gb.quicksum(self.variables.consumption[d,t] for d in self.DEMANDS)
                    - gb.quicksum(self.variables.generator_dispatch[g,t] for g in self.GENERATORS)
                    - gb.quicksum(self.variables.wind_turbines[w,t] for w in self.WINDTURBINES),
                    gb.GRB.EQUAL,
                    0, name='Balance equation') for t in self.TIMES}
        
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
        print("Generators:")
        print(self.results.profits_G)
        print("Wind turbines:")
        print(self.results.profits_W)
        print()
        print("Utility of demands: ")
        print(self.results.utilities)
        

def plot_SD_curve(ec, T):
    sort_offers = [('G100', 100, sum(ec.P_G_max.values()))]
    for g, offer_volume in ec.P_G_max.items():
        for ix, gen_data in enumerate(sort_offers):
            if gen_data[1] > ec.C_G_offer[g]:
                sort_offers.insert(ix, (g, ec.C_G_offer[g], offer_volume))
                break
    
    sort_offers = sort_offers[0:len(sort_offers)-1]
    plt.plot([0,sum(ec.P_W[T].values())], [0, 0], linewidth=1, color='blue', label='Supply Curve')
    point = np.array([sum(ec.P_W[T].values()), 0])

    for i in sort_offers:
        up_point = np.array([point[0], i[1]])
        right_point = np.array([point[0] + i[2], i[1]])
        plt.plot([point[0], up_point[0]], [point[1], up_point[1]], linewidth=1, color='blue')
        plt.plot([up_point[0], right_point[0]], [up_point[1], right_point[1]], linewidth=1, color='blue')
        point = right_point.copy()
    
    sort_bids = [('D100', 0, sum(ec.P_D[T].values()))]
    for d, bid_volume in ec.P_D[T].items():
        for ix, demand_data in enumerate(sort_bids):
            if demand_data[1] < ec.U_D[T][d]:
                sort_bids.insert(ix, (d, ec.U_D[T][d], bid_volume))
                break
    
    sort_bids = sort_bids[0:len(sort_bids)-1]
    
    plt.plot([0,sort_bids[0][2]], [sort_bids[0][1], sort_bids[0][1]], linewidth=1, color='orange', label='Demand Curve')
    point = np.array([sort_bids[0][2], sort_bids[0][1]])

    for i in sort_bids:
        down_point = np.array([point[0], i[1]])
        right_point = np.array([point[0] + i[2], i[1]])
        plt.plot([point[0], down_point[0]], [point[1], down_point[1]], linewidth=1, color='orange')
        plt.plot([down_point[0], right_point[0]], [down_point[1], right_point[1]], linewidth=1, color='orange')
        point = right_point.copy()
    
    plt.plot([point[0], point[0]], [point[1], 0], linewidth=1, color='orange')
    plt.title("Supply and Demand from 07:00 to 08:00")
    plt.xlabel("Quantity [MWh]")
    plt.ylabel("Price [$/MWh]")
    plt.axhline(ec.data.lambda_[T], color = 'black', linewidth=0.5, linestyle='--', label='Electricity Price')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    ec = EconomicDispatch(n_hours=1, ramping=False, battery=False, hydrogen=False)
    
    # ec.run()
    # ec.calculate_results()
    # ec.display_results()
    ec = EconomicDispatch(n_hours=24, ramping=True, battery=True, hydrogen=True)
    ec.run()
    ec.calculate_results()
    ec.display_results()
    
    plot_SD_curve(ec, 'T8')