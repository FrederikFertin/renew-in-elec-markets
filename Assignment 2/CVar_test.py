import gurobipy as gb
from gurobipy import GRB
import numpy as np
from scenario import DataInit
import matplotlib.pyplot as plt

"""
inputs: 
- P_max (nominal capacity)
- TIMES (set of times e.g., 'T1')
- SCENARIOS (set of scenarios e.g., 'S1')
- pi[w] (probability of each scenario)
- lambda_DA[t, w] (DA price at time t in scenario w)
- p_real[t, w] (realized production)
- imbalance_direction[t, w] (direction of imbalance at time t in scenario w)
"""



class expando(object):
    '''
        A small class which can have attributes set
    '''
    pass

class OfferingStrategy(DataInit):

    def __init__(self, price_scheme: str, alpha: float, beta: float):
        super().__init__()
        self.generate_scenarios(n_wind=20, n_price=20, n_balance=3, train_size=0.25, seed=42)
        self.data = expando()  # build data attributes
        self.variables = expando()  # build variable attributes
        self.constraints = expando()  # build constraint attributes
        self.results = expando()  # build results attributes
        self.price_scheme = price_scheme
        self.alpha = alpha
        self.beta = beta
        self._build_model()  # build gurobi model

    def _build_variables(self):
        # initialize variables
        self.variables.DA_dispatch = {
            t: self.model.addVar(lb=0, ub=self.P_max, name='dispatch at time {0}'.format(t)) for t in self.TIMES
        }
        self.variables.Delta = {
            (t, w): self.model.addVar(lb=-GRB.INFINITY, name='local imbalance {0}'.format(t))
            for t in self.TIMES for w in self.SCENARIOS
        }
        self.variables.Delta_UP = {
            (t, w): self.model.addVar(lb=0, name='up-regulation {0}'.format(t))
            for t in self.TIMES for w in self.SCENARIOS
        }
        self.variables.Delta_DOWN = {
            (t, w): self.model.addVar(lb=0, name='down-regulation {0}'.format(t))
            for t in self.TIMES for w in self.SCENARIOS
        }
        self.variables.zeta = self.model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name='zeta')
        self.variables.eta = {
            w : self.model.addVar(lb = 0, name = 'eta{0}'.format(w))
            for w in self.SCENARIOS
        }

    def _build_objective_function(self):

        DA_profits = gb.quicksum(self.pi[w] * self.lambda_DA[t,w] * self.variables.DA_dispatch[t]
                                 for w in self.SCENARIOS for t in self.TIMES
        )
        if self.price_scheme == 'one_price':
            UP_profits = gb.quicksum(
                self.pi[w] * 0.9 * self.lambda_DA[t, w] * self.variables.Delta_UP[t, w]
                for w in self.SCENARIOS for t in self.TIMES
            )
            DOWN_costs = gb.quicksum(
                self.pi[w] * 1.2 * self.lambda_DA[t, w] * self.variables.Delta_DOWN[t, w]
                for w in self.SCENARIOS for t in self.TIMES
            )
        elif self.price_scheme == 'two_price':
            UP_profits = gb.quicksum(
                self.pi[w] * 0.9**(self.imbalance_direction[t,w]) * self.lambda_DA[t,w] * self.variables.Delta_UP[t,w]
                for w in self.SCENARIOS for t in self.TIMES
            )
            DOWN_costs = gb.quicksum(
                self.pi[w] * 1.2**(1 - self.imbalance_direction[t,w]) * self.lambda_DA[t,w] * self.variables.Delta_DOWN[t,w]
                for w in self.SCENARIOS for t in self.TIMES
            )
        else:
            raise NotImplementedError
        CVar = self.variables.zeta - 1/(1-self.alpha) * gb.quicksum(self.pi[w] * self.variables.eta[w] for w in self.SCENARIOS)

        objective = (1-self.beta) * (DA_profits + UP_profits - DOWN_costs) + self.beta * CVar
        self.model.setObjective(objective, gb.GRB.MAXIMIZE)

    def _build_constraints(self):
        self.constraints.imbalance_constraints = {t: {w: self.model.addLConstr(
            self.variables.Delta[t,w],
            gb.GRB.EQUAL,
            self.p_real[t,w] - self.variables.DA_dispatch[t],
            name='Imbalance constraint')
            for w in self.SCENARIOS} for t in self.TIMES}
        self.constraints.regulation_constraints = {t: {w: self.model.addLConstr(
            self.variables.Delta[t, w],
            gb.GRB.EQUAL,
            self.variables.Delta_UP[t,w] - self.variables.Delta_DOWN[t,w],
            name='regulation constraint')
            for w in self.SCENARIOS} for t in self.TIMES}
        

        if self.price_scheme == 'one_price':
            self.constraints.eta_constraint = {w: self.model.addLConstr(
                - gb.quicksum(self.lambda_DA[t,w] * self.variables.DA_dispatch[t] + 
                              0.9 * self.lambda_DA[t, w] * self.variables.Delta_UP[t, w] -
                              1.2 * self.lambda_DA[t, w] * self.variables.Delta_DOWN[t, w]
                              for t in self.TIMES) + self.variables.zeta - self.variables.eta[w],
                gb.GRB.LESS_EQUAL,
                0, name = 'eta constraint') for w in self.SCENARIOS}
        elif self.price_scheme == 'two_price':
            self.constraints.eta_constraint = {w: self.model.addLConstr(
                - gb.quicksum(self.lambda_DA[t,w] * self.variables.DA_dispatch[t] + 
                              0.9**(self.imbalance_direction[t,w]) * self.lambda_DA[t,w] * self.variables.Delta_UP[t,w] -
                              1.2**(1 - self.imbalance_direction[t,w]) * self.lambda_DA[t,w] * self.variables.Delta_DOWN[t,w]
                              for t in self.TIMES) + self.variables.zeta - self.variables.eta[w],
                gb.GRB.LESS_EQUAL,
                0, name = 'eta constraint') for w in self.SCENARIOS}
        else:
            raise NotImplementedError

        

    def _build_model(self):
        # initialize optimization model
        self.model = gb.Model(name='Offering Strategy')

        self._build_variables()

        self.model.update()

        self._build_objective_function()

        self._build_constraints()

    def _save_data(self):
        # save objective value
        self.data.objective_value = self.model.ObjVal

        # save DA dispatch values
        self.data.DA_dispatch_values = {t:
            self.variables.DA_dispatch[t].x for t in self.TIMES
        }

        # save imbalance values
        self.data.Delta_values = {t: {w:
            self.variables.Delta[t,w].x for w in self.SCENARIOS} for t in self.TIMES
        }

        # save up-regulation values
        self.data.Delta_UP_values = {t: {w:
            self.variables.Delta_UP[t, w].x for w in self.SCENARIOS} for t in self.TIMES
        }

        # save down-regulation values
        self.data.Delta_DOWN_values = {t: {w:
            self.variables.Delta_DOWN[t, w].x for w in self.SCENARIOS} for t in self.TIMES
        }

        # save zeta value
        self.data.zeta = self.variables.zeta.x

        # save eta values
        self.data.eta_values = {w: self.variables.eta[w].x for w in self.SCENARIOS}

    def run_model(self):
        self.model.optimize()
        self._save_data()

    def calculate_results(self):
        self.results.DA_profits = {w:
            sum(self.lambda_DA[t, w] * self.data.DA_dispatch_values[t] for t in self.TIMES)
            for w in self.SCENARIOS
        }
        if self.price_scheme == 'one_price':
            self.results.BA_profits = {w:
                sum(0.9 * self.lambda_DA[t, w] * self.data.Delta_UP_values[t][w]
                - 1.2 * self.lambda_DA[t, w] * self.data.Delta_DOWN_values[t][w]
                for t in self.TIMES)
                for w in self.SCENARIOS
            }
        elif self.price_scheme == 'two_price':
            self.results.BA_profits = {w:
                sum(0.9**(self.imbalance_direction[t,w]) * self.lambda_DA[t, w] * self.data.Delta_UP_values[t][w]
                - 1.2**(1 - self.imbalance_direction[t,w]) * self.lambda_DA[t, w] * self.data.Delta_DOWN_values[t][w]
                for t in self.TIMES)
                for w in self.SCENARIOS
            }
        else:
            raise NotImplementedError
        
        self.results.expected_profit = sum(self.pi[w] * (self.results.DA_profits[w] +
                                                         self.results.BA_profits[w]) for w in self.SCENARIOS)
        
        self.results.CVaR = self.data.zeta - 1/(1-self.alpha) * sum(self.pi[w] * self.data.eta_values[w] for w in self.SCENARIOS)

    def display_results(self):
        print()
        print("-------------------   RESULTS  -------------------")
        print("Expected profit: ")
        print(self.results.expected_profit)
        print("CVar: ")
        print(self.results.CVaR)

if __name__ == '__main__':

    beta_values = np.linspace(0, 1, 21)
    expected_profits = []
    CVaRs = []
    for beta in beta_values:
        offering_strategy = OfferingStrategy(price_scheme='two_price', alpha = 0.95, beta = beta)
        offering_strategy.run_model()
        offering_strategy.calculate_results()
        expected_profits.append(offering_strategy.results.expected_profit)
        CVaRs.append(offering_strategy.results.CVaR)

    # plot results with both lines and points
    plt.plot(CVaRs, expected_profits, label='Expected profit', marker='o')
    plt.xlabel('CVaR')
    plt.ylabel('Expected profit')
    plt.show()
