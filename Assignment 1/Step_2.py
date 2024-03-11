import gurobipy as gb
from gurobipy import GRB

class CommonMethods:
    def add_ramping_constraints(self):
        self.constraints.ramping_dw = {(g,t):self.model.addConstr(
            self.variables.generator_dispatch[g,t] - self.variables.generator_dispatch[g,self.TIMES[n]],
            gb.GRB.GREATER_EQUAL,
            -self.P_R_DW[g]) for g in self.GENERATORS for n,t in enumerate(self.TIMES[1:])}
        self.constraints.ramping_up = {(g,t):self.model.addConstr(
            self.variables.generator_dispatch[g,t] - self.variables.generator_dispatch[g,self.TIMES[n]],
            gb.GRB.LESS_EQUAL,
            self.P_R_UP[g]) for g in self.GENERATORS for n,t in enumerate(self.TIMES[1:])}


    def add_battery_constraints(self):
        # soc constraint
        self.constraints.batt_soc = {(b,t):self.model.addLConstr(self.variables.battery_soc[b,t], 
        gb.GRB.EQUAL,
        self.variables.battery_soc[b,self.TIMES[n]] + self.batt_eta[b] * self.variables.battery_ch[b,t] - 1/self.batt_eta[b] * self.variables.battery_dis[b,t])
        for b in self.BATTERIES for n,t in enumerate(self.TIMES[1:])}
        # initializing soc constraint
        self.constraints.init_batt_soc = {(b):self.model.addLConstr(self.variables.battery_soc[b,self.TIMES[0]], 
            gb.GRB.EQUAL, 
            self.batt_init_soc[b] + self.batt_eta[b] * self.variables.battery_ch[b,self.TIMES[0]] - 1/self.batt_eta[b] * self.variables.battery_dis[b,self.TIMES[0]])
            for b in self.BATTERIES}
        # final soc constraint
        self.constraints.final_batt_soc = {(b):self.model.addLConstr(self.variables.battery_soc[b,self.TIMES[-1]],
            gb.GRB.GREATER_EQUAL,
            self.batt_init_soc[b])
            for b in self.BATTERIES}


    def add_hydrogen_constraints(self):
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
        

    def add_balance_constraints(self):
        balance_constraints = {}

        for t in self.TIMES: # Add one balance constraint for each hour
            # Contribution of generators
            generator_expr = gb.quicksum(self.variables.generator_dispatch[g,t] for g in self.GENERATORS)

            # Contribution of demands
            demand_expr = gb.quicksum(self.variables.consumption[d,t] for d in self.DEMANDS)

            # Contribution of wind farms
            wind_expr = gb.quicksum(self.variables.wind_turbines[w,t]
                            for w in self.WINDTURBINES)
            if self.H2:
                # Contribution of wind farms with hydrogen production
                wind_expr = gb.quicksum(self.variables.wind_turbines[w,t] - self.variables.hydrogen[w,t]
                            for w in self.WINDTURBINES)
            
            # Contribution of batteries
            batt_expr = 0
            if self.battery:
                batt_expr = gb.quicksum(self.variables.battery_ch[b,t] - self.variables.battery_dis[b,t] 
                            for b in self.BATTERIES)

            # Central balance constraint
            balance_constraints[t] = self.model.addLConstr(
                                        demand_expr - generator_expr - wind_expr + batt_expr,
                                        gb.GRB.EQUAL,
                                        0, name='Balance equation')
        return balance_constraints
        