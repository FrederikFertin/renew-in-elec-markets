import gurobipy as gb
import matplotlib.pyplot as plt
import random
import numpy as np

# TODO
    # - Fix ALSO-X
    # - Check CVar



class DataInit:
    def __init__(self, n_scenarios, length):
       
        self.SCENARIOS = range(n_scenarios)
        self.TIMES = range(length)
        self.scenarios = self._generate_load_profiles(n_scenarios, length)
        
    def _generate_load_profiles(self, n, l):
        load_profiles = []
        
        for _ in range(n):
            load_profile = [random.randint(300, 500)]  # Start with an initial load
            
            for _ in range(1, l):
                load = load_profile[-1] + random.randint(-25, 25)  # Randomly increase or decrease the load by 1
                if load > 500:
                    load = 500
                elif load < 200:
                    load = 200
                load_profile.append(load)
            
            load_profiles.append(load_profile)
        
        return np.array(load_profiles).T
    

    
class expando(object):
    '''
        A small class which can have attributes set
    '''
    pass

class ancillary_service(DataInit):

    def __init__(self, solution_technique: str, n_scenarios : int, length : int, eps : float):
        super().__init__(n_scenarios, length)
        self.data = expando()  # build data attributes
        self.variables = expando()  # build variable attributes
        self.constraints = expando()  # build constraint attributes
        self.results = expando()  # build results attributes
        self.solution_technique = solution_technique
        self.eps = eps
        self.q = self.eps * len(self.SCENARIOS) * len(self.TIMES)
        self._build_model()  # build gurobi model
        
    


    def _build_variables(self):
        # initialize variables
        self.variables.c_up = self.model.addVar(lb=0, name='up-regulation capacity')

        if self.solution_technique == 'MILP':
            self.variables.y = {
                (t, w): self.model.addVar(vtype=gb.GRB.BINARY, lb=0, name='violation {0}'.format(t)) for t in self.TIMES for w in self.SCENARIOS
            }
        elif self.solution_technique == 'ALSO-X':
            self.variables.y = {
                (t, w): self.model.addVar(lb=0, ub = 1, name='violation {0}'.format(t)) for t in self.TIMES for w in self.SCENARIOS
            }
        elif self.solution_technique == 'CVar':
            self.variables.beta = self.model.addVar(ub=0, name='Weight')

            self.variables.xi = {
                (t, w): self.model.addVar(lb=0, name='xi {0}'.format(t)) for t in self.TIMES for w in self.SCENARIOS
            }

    def _build_objective_function(self):
        self.model.setObjective(self.variables.c_up, gb.GRB.MAXIMIZE)


    def _build_constraints(self):
        if (self.solution_technique == 'MILP' or self.solution_technique == 'ALSO-X'):
            self.constraints.violation_constraints = {t: {w: self.model.addConstr(
                self.variables.c_up - self.scenarios[t,w] <= self.variables.y[t, w] * 500, name='violation constraint {0}'.format(t))  # Check big M.
                for w in self.SCENARIOS} for t in self.TIMES
            }
            self.constraints.violation_limit = self.model.addConstr(
                gb.quicksum(self.variables.y[t, w] for t in self.TIMES for w in self.SCENARIOS) <= self.eps * len(self.SCENARIOS) * len(self.TIMES)) 
        
        elif self.solution_technique == 'CVar':
            self.constraints.violation_constraints = {t: {w: self.model.addConstr(
                self.variables.c_up - self.scenarios[t,w] <= self.variables.xi[t, w], name='violation constraint {0}'.format(t)) 
                for w in self.SCENARIOS} for t in self.TIMES
            }
            self.constraints.violation_limit = self.model.addConstr(1/(len(self.SCENARIOS) * len(self.TIMES)) * gb.quicksum(self.variables.xi[t, w] for t in self.TIMES for w in self.SCENARIOS) 
                                                               <= (1-self.eps) * self.variables.beta)
            self.constraints.cvar_constraint = {t: {w: self.model.addConstr(
                self.variables.beta <= self.variables.xi[t, w], name='cvar constraint {0}'.format(t)) for w in self.SCENARIOS} for t in self.TIMES              
            }
        else:
            raise ValueError('Invalid solution technique')
    
    def _also_X(self):
        q_underline = 0
        q_overline = self.q
        while q_overline - q_underline > 10E-5:
            self.q = (q_underline + q_overline) / 2
            self.model.optimize()
            self._save_data()
            self.P = sum((self.data.y[t, w] < 10E-6) for t in self.TIMES for w in self.SCENARIOS)/ (len(self.TIMES) * len(self.SCENARIOS))
            if self.P >= (1-self.eps):
                q_underline = self.q
            else:
                q_overline = self.q
            self._build_model()
            
    
    def _build_model(self):
        # initialize optimization model
        self.model = gb.Model(name='Ancillary Service')

        self._build_variables()

        self.model.update()

        self._build_objective_function()

        self._build_constraints()

    def _save_data(self):
        self.data.c_up = self.variables.c_up.X
        
        if (self.solution_technique == 'MILP' or self.solution_technique == 'ALSO-X'):
            self.data.y = {(t, w): self.variables.y[t, w].X for t in self.TIMES for w in self.SCENARIOS}
        elif self.solution_technique == 'CVar':
            self.data.beta = self.variables.beta.X
            self.data.xi = {(t, w): self.variables.xi[t, w].X for t in self.TIMES for w in self.SCENARIOS}
            
    def run_model(self):
        if self.solution_technique == 'ALSO-X':
            self._also_X()
        self.model.optimize()
        self._save_data()

    def calculate_results(self):
        ...
        
    def display_results(self):
        print()
        print("-------------------   RESULTS  -------------------")
        print("Bid quantity:")
        print(self.data.c_up)
        

if __name__ == '__main__':
 
    
    #anc = ancillary_service('CVar', 50, 60, 0.1)
    #anc.run_model()
    
    anc = ancillary_service('MILP', 50, 60, 0.1)
    #anc.run_model()

    #anc = ancillary_service('ALSO-X', 50, 60, 0.1)
    anc.run_model()
    c = anc.data.c_up
    q = np.repeat(c, 60)
    t = anc.scenarios
    plt.plot(t, alpha = 0.2)
    plt.plot(q, 'r')
    plt.show()