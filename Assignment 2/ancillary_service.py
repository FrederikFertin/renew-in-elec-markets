import numpy as np
import gurobipy as gb



class expando(object):
    '''
        A small class which can have attributes set
    '''
    pass

class ancillary_service():

    def __init__(self, solution_technique: str):
        self.scenarios = [self._generate_load_profiles for _ in range(200)]
        self.SCENARIOS = range(200)
        self.TIMES = range(60)
        self.data = expando()  # build data attributes
        self.variables = expando()  # build variable attributes
        self.constraints = expando()  # build constraint attributes
        self.results = expando()  # build results attributes
        self.solution_technique = solution_technique
        self._build_model()  # build gurobi model

    def _generate_load_profiles(n=60):
        x0 = np.random.randint(200, 500)
        eps = np.random.normal(0, 50, n)

        x = np.zeros(n)
        x[0] = x0
        for i in range(1,n):
            x[i] = x[i-1] + eps[i]
            if x[i] > 500:
                x[i] = 500
            elif x[i] < 200:
                x[i] = 200
            if x[i] - x[i-1] > 25:
                x[i] = x[i-1] + 25
            elif x[i] - x[i-1] < -25:
                x[i] = x[i-1] - 25
        return x


    def _build_variables(self):
        # initialize variables
        self.variables.c_up = {self.model.addVar(lb=0, name='up-regulation capacity')}

        if self.solution_technique == 'MILP':
            self.variables.y = {
                (t, w): self.model.addVar(vtype=gp.GRB.BINARY, lb=0, name='violation {0}'.format(t)) for t in self.TIMES for w in self.SCENARIOS
            }
        elif self.solution_technique == 'ALSO-X':
            self.variables.y = {
                (t, w): self.model.addVar(lb=0, name='violation {0}'.format(t)) for t in self.TIMES for w in self.SCENARIOS
            }
        elif self.solution_technique == 'CVar':
            self.variables.beta = {
                self.model.addVar(ub=0, name='Weight')
            }
            self.variables.xi = {
                (t, w): self.model.addVar(lb=0, name='xi {0}'.format(t)) for t in self.TIMES for w in self.SCENARIOS
            }

    def _build_objective_function(self):
        self.model.setObjective(gb.quicksum(self.variables.c_up[t] for t in self.TIMES), gb.GRB.MAXIMIZE)


    def _build_constraints(self):
        if self.solution_technique == 'MILP':
            self.constraints.violation_constraints = {t: {w: self.model.addConstr(
                self.c_up - self.scenarios[w][t] <= 500 * self.variables.y[t, w], name='violation constraint {0}'.format(t)) 
                for w in self.SCENARIOS} for t in self.TIMES
            }
            self.constraints.violation_limit = self.model.addConstr(gb.quicksum(self.variables.y[t, w] for t in self.TIMES for w in self.SCENARIOS) <= 6 ) # Hardcoded, fix this

    def _build_model(self):
        # initialize optimization model
        self.model = gb.Model(name='Ancillary Service')

        self._build_variables()

        self.model.update()

        self._build_objective_function()

        self._build_constraints()

    def _save_data(self):
        self.data.c_up = self.variables.c_up.X

    def run_model(self):
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
    anc = ancillary_service('MILP')
    anc.run_model()
    anc.display_results()
