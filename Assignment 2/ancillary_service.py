import gurobipy as gb
import matplotlib.pyplot as plt
import random
import numpy as np
from typing import Union

# TODO
    # - Fix ALSO-X
    # - Check CVar
    # - Split into training and testing data

random.seed(42)
np.random.seed(42)


class DataInit:
    def __init__(self):
       
        self.SCENARIOS = range(200)
        self.TIMES = range(60)
        self.scenarios = self._generate_load_profiles()
        
    def _generate_load_profiles(self):
        load_profiles = []
        
        for _ in self.SCENARIOS:
            load_profile = [random.randint(300, 400)]  # Start with an initial load
            
            for _ in range(1, 60):
                load = load_profile[-1] + random.randint(-25, 25)  # Randomly increase or decrease the load by up to 25
                if load > 500:
                    load = 500
                elif load < 200:
                    load = 200
                load_profile.append(load)
            
            load_profiles.append(load_profile)
        
        return np.array(load_profiles).T
    
    def create_train_test_split(self, train_size: int = 50, k: Union[int, None] = None):
        if k is None:
            self.train_scenarios = self.scenarios[:,:train_size]
            self.test_scenarios = self.scenarios[:,train_size:]
        else:
            self.train_scenarios = self.scenarios[:,k * train_size:(k + 1) * train_size]
            self.test_scenarios = self.scenarios[:,:k*train_size]
            if train_size*(k+1) < len(self.scenarios):
                self.test_scenarios.extend(self.scenarios[:,(k+1)*train_size:])

        self.n_scenarios = len(self.train_scenarios[0])
        self.SCENARIOS = range(self.n_scenarios)
        self.pi = np.ones(self.n_scenarios)/self.n_scenarios
    

    
class expando(object):
    '''
        A small class which can have attributes set
    '''
    pass

class ancillary_service(DataInit):

    def __init__(self, solution_technique: str, eps : float):
        super().__init__()
        self.create_train_test_split()
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
                (t, w): self.model.addVar(vtype=gb.GRB.BINARY, name='violation {0}'.format(t)) for t in self.TIMES for w in self.SCENARIOS
            }
        elif self.solution_technique == 'ALSO-X':
            self.variables.y = {
                (t, w): self.model.addVar(lb=0, ub = 1, name='violation {0}'.format(t)) for t in self.TIMES for w in self.SCENARIOS
            }
        elif self.solution_technique == 'CVaR':
            self.variables.beta = self.model.addVar(ub=0, name='Weight')

            self.variables.zeta = {
                (t, w): self.model.addVar(lb=0, name='zeta {0}'.format(t)) for t in self.TIMES for w in self.SCENARIOS
            }

    def _build_objective_function(self):
        self.model.setObjective(self.variables.c_up, gb.GRB.MAXIMIZE)


    def _build_constraints(self):
        if (self.solution_technique == 'MILP' or self.solution_technique == 'ALSO-X'):
            self.constraints.violation_constraints = {t: {w: self.model.addConstr(
                self.variables.c_up - self.train_scenarios[t,w] <= self.variables.y[t, w] * 300, name='violation constraint {0}'.format(t))  # Check big M.
                for w in self.SCENARIOS} for t in self.TIMES
            }
            self.constraints.violation_limit = self.model.addConstr(
                gb.quicksum(self.variables.y[t, w] for t in self.TIMES for w in self.SCENARIOS) <= self.eps * len(self.SCENARIOS) * len(self.TIMES)) 
        
        elif self.solution_technique == 'CVaR':
            self.constraints.violation_constraints = {t: {w: self.model.addConstr(
                self.variables.c_up - self.train_scenarios[t,w] <= self.variables.zeta[t, w], name='violation constraint {0}'.format(t)) 
                for w in self.SCENARIOS} for t in self.TIMES
            }
            self.constraints.violation_limit = self.model.addConstr(
                1/(len(self.SCENARIOS) * len(self.TIMES)) * gb.quicksum(self.variables.zeta[t, w] for t in self.TIMES for w in self.SCENARIOS) 
                                                               <= (1-self.eps) * self.variables.beta)
            self.constraints.cvar_constraint = {t: {w: self.model.addConstr(
                self.variables.beta <= self.variables.zeta[t, w], name='CVaR constraint {0}'.format(t)) for w in self.SCENARIOS} for t in self.TIMES              
            }
        else:
            raise ValueError('Invalid solution technique')
    
    def _also_X(self):
        q_underline = 0
        q_overline = self.eps * len(self.SCENARIOS)**2
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
        elif self.solution_technique == 'CVaR':
            self.data.beta = self.variables.beta.X
            self.data.zeta = {(t, w): self.variables.zeta[t, w].X for t in self.TIMES for w in self.SCENARIOS}
            
    def run_model(self):
        if self.solution_technique == 'ALSO-X':
            self._also_X()
        self.model.optimize()
        self._save_data()

        
    def display_results(self):
        print()
        print("-------------------   RESULTS  -------------------")
        print("Bid quantity:")
        print(self.data.c_up)
        

def out_of_sample_analysis(anc):
    c_up = anc.data.c_up
    bid = np.repeat(c_up, 60)
    fig, axs = plt.subplots(2, figsize = (8, 8), sharex=True)
    fig.subplots_adjust(hspace=0.2, wspace=0.2)

    train = anc.train_scenarios
    axs[0].plot(train, color = 'b', alpha = 0.1)
    axs[0].plot(bid, 'r')
    axs[0].set_title('In-sample data')
    axs[0].set_ylabel('Load (kW)')
    
    test = anc.test_scenarios
    axs[1].plot(test, color  = 'b', alpha = 0.1)
    axs[1].plot(bid, 'r', label = 'Bid quantity')
    axs[1].set_title('Out-of-sample data')
    axs[1].set_ylabel('Load (kW)')
    axs[1].legend()
    plt.show()
    
    print("Violations of reserve capacity bid:", round(sum(sum(test < c_up))/(np.size(test))*100,2), " %")
    print("Average shortfall:", -(test[test < c_up]-c_up).mean().round(2), " kW")


def p90_variations(eps):
    bids = []
    violations = []
    shortfalls = []
    for e in eps:
        anc = ancillary_service('MILP',  e)
        anc.run_model()
        c_up = anc.data.c_up
        bids.append(c_up)
        violations.append(sum(sum(anc.test_scenarios < c_up))/(np.size(anc.test_scenarios))*100)
        shortfalls.append(-(anc.test_scenarios[anc.test_scenarios < c_up]-c_up).mean())
    
    # Plot bids and violations as function of epsilon in one plot with dual y-axis
    fig, axs = plt.subplots(3, figsize = (10, 10))
    axs[0].plot(eps*100, bids, 'r', marker = 'o')
    axs[0].set_ylabel('Bid quantity')
    axs[1].plot(eps*100, violations, 'b', marker = 'o')
    axs[1].set_ylabel('Violations (%)')
    axs[2].plot(eps*100, shortfalls, 'g', marker = 'o')
    axs[2].set_ylabel('Average shortfall (kW)')
    axs[2].set_xlabel('Epsilon (%)')
    plt.show()

if __name__ == '__main__':
 
    # 2.1
    anc = ancillary_service('CVaR', 0.1)
    #anc = ancillary_service('MILP',  0.1)
    #anc = ancillary_service('ALSO-X',  0.1)
    
    anc.run_model()
    
    # 2.2
    out_of_sample_analysis(anc)
    
    
    # 2.3
    eps = np.arange(0, 0.21, 0.01)
    #p90_variations(eps)
    