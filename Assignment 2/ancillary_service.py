import gurobipy as gb
import matplotlib.pyplot as plt
import random
import numpy as np
from time import time
from typing import Union

class DataInit:
    def __init__(self):
       
        self.SCENARIOS = range(200)
        self.TIMES = range(60)
        self.scenarios = self._generate_load_profiles()
        
    def _generate_load_profiles(self):
        load_profiles = []
        
        for _ in self.SCENARIOS:
            # Start with an initial load
            load_profile = [random.randint(300, 400)]  
            
            for _ in range(1, 60):
                # Randomly increase or decrease the load by up to 25
                load = load_profile[-1] + random.randint(-25, 25)  
                # Ensure the load is within the bounds of 200 and 500
                if load > 500:
                    load = 500
                elif load < 200:
                    load = 200
                load_profile.append(load)
            
            load_profiles.append(load_profile)
        
        return np.array(load_profiles).T
    
    def create_train_test_split(self, train_size: int = 50, k: Union[int, None] = None):
        # Split the scenarios into training and testing sets
        if k is None:
            self.train_scenarios = self.scenarios[:,:train_size]
            self.test_scenarios = self.scenarios[:,train_size:]
        else:
            self.train_scenarios = self.scenarios[:,k * train_size:(k + 1) * train_size]
            self.test_scenarios = self.scenarios[:,:k*train_size]
            if train_size*(k+1) < len(self.scenarios):
                self.test_scenarios.extend(self.scenarios[:,(k+1)*train_size:])

        # Set the number of scenarios and the scenario index
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
        self.create_train_test_split() # generate scenarios and split into train and test
        self.data = expando()  # build data attributes
        self.variables = expando()  # build variable attributes
        self.constraints = expando()  # build constraint attributes
        self.results = expando()  # build results attributes
        self.solution_technique = solution_technique # MILP or CVaR
        self.eps = eps # Allowed violations (%)
        self.q = self.eps * len(self.SCENARIOS) * len(self.TIMES) # Allowed violations (abs)
        self._build_model()  # build gurobi model

    def _build_variables(self):
        # initialize variables
        self.variables.c_up = self.model.addVar(lb=0, name='up-regulation capacity')

        if self.solution_technique == 'MILP':
            self.variables.y = {
                (t, w): self.model.addVar(vtype=gb.GRB.BINARY, name='violation {0}'.format(t)) for t in self.TIMES for w in self.SCENARIOS
            }
        elif self.solution_technique == 'CVaR':
            self.variables.beta = self.model.addVar(lb = -gb.GRB.INFINITY, ub=0, name='Weight')

            self.variables.zeta = {
                (t, w): self.model.addVar(lb=-gb.GRB.INFINITY, name='zeta {0}'.format(t)) for t in self.TIMES for w in self.SCENARIOS
            }

    def _build_objective_function(self):
        self.model.setObjective(self.variables.c_up, gb.GRB.MAXIMIZE)

    def _build_constraints(self):
        if (self.solution_technique == 'MILP'):
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

    def _build_model(self):
        # initialize optimization model
        self.model = gb.Model(name='Ancillary Service')

        self._build_variables()

        self.model.update()

        self._build_objective_function()

        self._build_constraints()

    def _save_data(self):
        self.data.c_up = self.variables.c_up.X
        
        if (self.solution_technique == 'MILP'):
            self.data.y = {(t, w): self.variables.y[t, w].X for t in self.TIMES for w in self.SCENARIOS}
        elif self.solution_technique == 'CVaR':
            self.data.beta = self.variables.beta.X
            self.data.zeta = {(t, w): self.variables.zeta[t, w].X for t in self.TIMES for w in self.SCENARIOS}
        
        self.data.train_violations = sum(sum(self.train_scenarios < self.data.c_up))/(np.size(self.train_scenarios))
        self.data.test_violations = sum(sum(self.test_scenarios < self.data.c_up))/(np.size(self.test_scenarios))
        self.data.average_shortfall = -(self.test_scenarios[self.test_scenarios < self.data.c_up]-self.data.c_up).mean()
        self.data.running_time = time() - self.time

    def run_model(self):
        self.time = time()
        self.model.optimize()
        self._save_data()

    def display_results(self):
        print()
        print("-------------------   RESULTS  -------------------")
        print("Running time:")
        print(round(self.data.running_time, 2), "s")
        print("Bid quantity:")
        print(round(self.data.c_up, 2), "kW")
        print("Violations in training data:")
        print(round(self.data.train_violations*100, 2), "%")
        print("Violations in test data:")
        print(round(self.data.test_violations*100, 2), "%")
        print("Average shortfall in test data:")
        print(round(self.data.average_shortfall, 2), "kW")
        

def plot_profiles(anc):
    # Plot in-sample and out-of-sample data with bid quantity
    c_up = anc.data.c_up
    bid = np.repeat(c_up, 60)
    fig, axs = plt.subplots(2, figsize = (8, 8), sharex=True)
    fig.subplots_adjust(hspace=0.2, wspace=0.2)

    train = anc.train_scenarios
    axs[0].plot(train, color = 'b', alpha = 0.1)
    axs[0].plot(bid, 'r')
    axs[0].set_title('In-sample data')
    axs[0].set_ylabel('Load [kW]')
    
    test = anc.test_scenarios
    axs[1].plot(test, color  = 'b', alpha = 0.1)
    axs[1].plot(bid, 'r', label = 'Bid quantity')
    axs[1].set_title('Out-of-sample data')
    axs[1].set_ylabel('Load [kW]')
    axs[1].set_xlabel('Time [min]')
    axs[1].legend()
    plt.show()

def p90_variations(eps):
    # Calculate bids, violations and shortfalls for different epsilon values
    bids = []
    violations = []
    shortfalls = []
    for e in eps:
        anc = ancillary_service('MILP',  e)
        anc.run_model()
        c_up = anc.data.c_up
        bids.append(c_up)
        violations.append(anc.data.test_violations*100)
        shortfalls.append(anc.data.average_shortfall)
    
    shortfalls = np.nan_to_num(np.array(shortfalls))
    
    # Plot bids and violations as function of epsilon in one plot with dual y-axis
    fig, axs = plt.subplots(3, figsize = (10, 10), sharex=True)
    axs[0].plot(eps*100, bids, 'r', marker = 'o')
    axs[0].set_xticks(eps*100)
    axs[0].set_ylabel('Bid quantity [kW]')
    axs[1].plot(eps*100, violations, 'b', marker = 'o')
    axs[1].set_ylabel('Violations [%]')
    axs[2].plot(eps*100, shortfalls, 'g', marker = 'o')
    axs[2].set_ylabel('Average shortfall [kW]')
    axs[2].set_xlabel('Epsilon [%]')
    plt.show()


if __name__ == '__main__':
    random.seed(42)
    np.random.seed(42)

    # 2.1
    anc_cvar = ancillary_service('CVaR', 0.1)
    anc_cvar.run_model()
    anc_cvar.display_results()

    anc_milp = ancillary_service('MILP',  0.1)
    anc_milp.run_model()
    anc_milp.display_results()

    # 2.2
    plot_profiles(anc_cvar)
    plot_profiles(anc_milp)
    
    # 2.3
    eps = np.arange(0, 0.21, 0.01)
    p90_variations(eps)
    