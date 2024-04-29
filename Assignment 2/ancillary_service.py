import numpy as np




class expando(object):
    '''
        A small class which can have attributes set
    '''
    pass

class ancillary_service():

    def __init__(self, solution_technique: str):
        self.scenarios = [self._generate_load_profiles() for i in range(200)]
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
        self.variables.c_up = {
            t: self.model.addVar(lb=0, name='up-regulation capacity {0}'.format(t)) for t in self.TIMES
        }


    def _build_objective_function(self):
        ...

    def _build_constraints(self):
        ...

    def _build_model(self):
        # initialize optimization model
        self.model = gb.Model(name='Ancillary Service')

        self._build_variables()

        self.model.update()

        self._build_objective_function()

        self._build_constraints()

    def _save_data(self):
        ...

    def run_model(self):
        self.model.optimize()
        self._save_data()

    def calculate_results(self):
        ...
        
    def display_results(self):
        print()
        print("-------------------   RESULTS  -------------------")
        print("Expected profit: ")
        print(round(self.data.objective_value, 2))
        print("Optimal bid: ")
        print({t: round(self.data.DA_dispatch_values[t], 2) for t in self.TIMES})

