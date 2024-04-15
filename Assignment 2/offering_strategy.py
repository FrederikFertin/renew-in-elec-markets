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

    def __init__(self, price_scheme: str):
        super().__init__()
        self.generate_scenarios(n_wind=20, n_price=20, n_balance=3, train_size=0.25, seed=42)
        self.data = expando()  # build data attributes
        self.variables = expando()  # build variable attributes
        self.constraints = expando()  # build constraint attributes
        self.results = expando()  # build results attributes
        self.price_scheme = price_scheme
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

        objective = DA_profits + UP_profits - DOWN_costs
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

    def run_model(self):
        self.model.optimize()
        self._save_data()

    def calculate_results(self):
        self.results.DA_profits = {w:
            self.lambda_DA[t, w] * self.data.DA_dispatch_values[t]
            for t in self.TIMES for w in self.SCENARIOS
        }
        if self.price_scheme == 'one_price':
            self.results.BA_profits = {w:
                0.9 * self.lambda_DA[t, w] * self.data.Delta_UP_values[t][w]
                - 1.2 * self.lambda_DA[t, w] * self.data.Delta_DOWN_values[t][w]
                for t in self.TIMES for w in self.SCENARIOS
            }
        elif self.price_scheme == 'two_price':
            self.results.BA_profits = {w:
                0.9**(self.imbalance_direction[t,w]) * self.lambda_DA[t, w] * self.data.Delta_UP_values[t][w]
                - 1.2**(1 - self.imbalance_direction[t,w]) * self.lambda_DA[t, w] * self.data.Delta_DOWN_values[t][w]
                for t in self.TIMES for w in self.SCENARIOS
            }
        else:
            raise NotImplementedError
        self.results.total_profits = {w:
            self.results.DA_profits[w] + self.results.BA_profits[w]
            for w in self.SCENARIOS
        }
        
    def display_results(self):
        print()
        print("-------------------   RESULTS  -------------------")
        print("Expected profit: ")
        print(round(self.data.objective_value, 2))
        print("Optimal bid: ")
        print({t: round(self.data.DA_dispatch_values[t], 2) for t in self.TIMES})

if __name__ == '__main__':
    offering_strategy = OfferingStrategy(price_scheme='two_price')
    offering_strategy.run_model()
    offering_strategy.calculate_results()
    offering_strategy.display_results()

    # Two types of plots to visualize the results
    # Plot mosaic of histograms of total profit and broken down into DA and BA profits
    fig, ax = plt.subplot_mosaic([["upper", "upper"],["lower right", "lower left"]], layout = "constrained")
    ax["upper"].hist(list(offering_strategy.results.total_profits.values()), bins=20, color='b', alpha=0.7, label='Total profits')
    ax["upper"].set_title('Total profits')
    ax["lower left"].hist(list(offering_strategy.results.BA_profits.values()), bins=20, color='g', alpha=0.7, label='BA profits')
    ax["lower left"].set_title('BA profits')
    ax["lower right"].hist(list(offering_strategy.results.DA_profits.values()), bins=20, color='r', alpha=0.7, label='DA profits')
    ax["lower right"].set_title('DA profits')
    plt.show()
    
    # Plot the profit histograms in three subplots above each other with shared x-axis
    fig, ax = plt.subplots(3, 1, sharex=True)
    ax[0].hist(list(offering_strategy.results.total_profits.values()), bins=20, color='b', alpha=0.7, label='Total profits')
    ax[1].hist(list(offering_strategy.results.BA_profits.values()), bins=20, color='g', alpha=0.7, label='BA profits')
    ax[2].hist(list(offering_strategy.results.DA_profits.values()), bins=20, color='r', alpha=0.7, label='DA profits')
    plt.show()