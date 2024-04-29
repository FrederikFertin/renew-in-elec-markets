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

    def __init__(
            self,
            risk_type: str,
            price_scheme: str,
            alpha: float = 0.9,
            beta: float = 0.5,
            train_size: int = 950,
            k: int | None = None,
    ):
        super().__init__()
        self.generate_scenarios(n_wind=20, n_price=20, n_balance=3, seed=42)
        self.create_train_test_split(train_size=train_size, k=k)
        self.data = expando()  # build data attributes
        self.variables = expando()  # build variable attributes
        self.constraints = expando()  # build constraint attributes
        self.results = expando()  # build results attributes
        self.risk_type = risk_type
        self.price_scheme = price_scheme
        self.alpha = alpha
        if risk_type == 'neutral':
            self.beta = 0
        elif risk_type == 'averse':
            self.beta = beta
        else:
            raise NotImplementedError
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
        if self.risk_type == 'averse':
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
        if self.risk_type == 'averse':
            CVar = self.variables.zeta - 1/(1-self.alpha) * gb.quicksum(self.pi[w] * self.variables.eta[w] for w in self.SCENARIOS)
        else:
            CVar = 0

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
        
        if self.risk_type == 'averse':
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

        if self.risk_type == 'averse':
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
        
        self.results.total_profits = {w:
            self.results.DA_profits[w] + self.results.BA_profits[w]
            for w in self.SCENARIOS
        }

        self.results.expected_profit = sum(self.pi[w] * (self.results.DA_profits[w] +
                                                         self.results.BA_profits[w]) for w in self.SCENARIOS)
        if self.risk_type == 'averse':
            self.results.CVaR = self.data.zeta - 1/(1-self.alpha) * sum(self.pi[w] * self.data.eta_values[w] for w in self.SCENARIOS)

    def display_results(self):
        print()
        print("-------------------   RESULTS  -------------------")
        print("Expected profit: ")
        print(self.results.expected_profit)
        print("CVar: ")
        print(self.results.CVaR)

    def calculate_oos_profits(self):
        self.results.oos_profits = []

        for scenario in self.test_scenarios:
            wind = scenario['wind']
            lambda_DA = scenario['lambda']
            imbalance = scenario['system_balance']
            DA_profits = sum(
                lambda_DA[t] * self.data.DA_dispatch_values[t] for t in self.TIMES)
            if self.price_scheme == 'one_price':
                BA_profits = sum(0.9 * lambda_DA[t] * max(wind[t] - self.data.DA_dispatch_values[t], 0)
                                 - 1.2 * lambda_DA[t] * max(self.data.DA_dispatch_values[t] - wind[t], 0)
                                 for t in self.TIMES)
            elif self.price_scheme == 'two_price':
                BA_profits = sum(
                    0.9 ** (imbalance[t]) * lambda_DA[t] * max(wind[t] - self.data.DA_dispatch_values[t], 0)
                    - 1.2 ** (1 - imbalance[t]) * lambda_DA[t] * max(self.data.DA_dispatch_values[t] - wind[t],0)
                    for t in self.TIMES)
            else:
                raise NotImplementedError
            Total_profits = DA_profits + BA_profits
            self.results.oos_profits.append(Total_profits)

        average_oos_profit = np.mean(self.results.oos_profits)
        print('Average out-of-sample profit: ', average_oos_profit)
        print('Average in-sample profit: ', self.results.expected_profit)

    def plot_is_profits(self):
        in_sample_profits = list(self.results.total_profits.values())
        plt.hist(in_sample_profits, bins=20, density=True)
        plt.xlabel('In-sample profits')
        plt.ylabel('Frequency')
        plt.show()

    def plot_oos_profits(self):
        plt.hist(self.results.oos_profits, bins=20, density=True)
        plt.xlabel('Out-of-sample profits')
        plt.ylabel('Frequency')
        plt.show()


def plot_beta_vs_cvar(beta_values: list, price_scheme: str):
    expected_profits = []
    CVaRs = []
    for beta in beta_values:
        offering_strategy = OfferingStrategy(risk_type='averse', price_scheme=price_scheme, alpha=0.9, beta=beta)
        offering_strategy.run_model()
        offering_strategy.calculate_results()
        expected_profits.append(offering_strategy.results.expected_profit)
        CVaRs.append(offering_strategy.results.CVaR)

    # plot results with both lines and points
    plt.plot(CVaRs, expected_profits, label='Expected profit', marker='o')
    for ix, beta in enumerate(beta_values):
        plt.annotate(round(float(beta), 2), (CVaRs[ix], expected_profits[ix]))
    plt.xlabel('CVaR')
    plt.ylabel('Expected profit')
    plt.show()

def plot_train_size_vs_profit_diff(beta: float):
    train_sizes = np.linspace(100, 1100, 11).astype(int)
    profit_diffs = []

    for train_size in train_sizes:
        offering_strategy = OfferingStrategy(risk_type='averse', price_scheme='one_price', alpha = 0.9, beta = beta, train_size=train_size)
        offering_strategy.run_model()
        offering_strategy.calculate_results()
        offering_strategy.calculate_oos_profits()

        avg_is_profits = np.mean(list(offering_strategy.results.total_profits.values()))
        avg_oos_profits = np.mean(offering_strategy.results.oos_profits)

        profit_diffs.append(abs(avg_is_profits - avg_oos_profits))

    plt.title("Train size vs profit differences")
    plt.plot(train_sizes, profit_diffs)
    plt.xlabel("Train size")
    plt.ylabel("Absolute profit difference")
    plt.show()

def plot_train_size_vs_profit_diff_k_fold(beta: float):
    train_sizes = [100, 200, 300, 400, 600]
    profit_diffs = []

    for train_size in train_sizes:
        avg_is_profits = []
        avg_oos_profits = []
        for k in range(1200 // train_size):
            offering_strategy = OfferingStrategy(risk_type='averse', price_scheme='one_price', alpha=0.9, beta=beta,
                                                 train_size=train_size, k=k)
            offering_strategy.run_model()
            offering_strategy.calculate_results()
            offering_strategy.calculate_oos_profits()

            avg_is_profits.append(np.mean(list(offering_strategy.results.total_profits.values())))
            avg_oos_profits.append(np.mean(offering_strategy.results.oos_profits))

        profit_diffs.append(abs(np.mean(avg_is_profits) - np.mean(avg_oos_profits)))

    plt.title("Train size vs profit differences using k-fold cross validation")
    plt.plot(train_size, profit_diffs)
    plt.xlabel("Train size")
    plt.ylabel("Absolute profit difference")
    plt.show()

if __name__ == '__main__':
    beta = 0.25
    plot_train_size_vs_profit_diff_k_fold(beta=beta)

    ##### ---------- Step: Test the model with two-price scheme ---------- #####
    # Step 1.3: Test the model with two-price scheme
    beta_values = np.linspace(0, 1, 21)
    # plot_beta_vs_cvar(beta_values, 'two_price')

    ### For two-price scheme the optimal beta is decided to be 0.3
    beta = 0.25
    offering_strategy = OfferingStrategy(risk_type='averse', price_scheme='two_price', alpha = 0.9, beta=beta)
    offering_strategy.run_model()
    offering_strategy.calculate_results()

    # Calculate out-of-sample profits
    offering_strategy.calculate_oos_profits()
    offering_strategy.plot_oos_profits()

    ##### ---------- Step: Test the model with one-price scheme ---------- #####
    # Step 1.3: Test the model with two-price scheme
    beta_values = np.linspace(0, 1, 21)
    plot_beta_vs_cvar(beta_values, 'one_price')

    ### For one-price scheme the optimal beta is decided to be 0.3
    beta = 0.4
    offering_strategy = OfferingStrategy(risk_type='averse', price_scheme='one_price', alpha = 0.9, beta = beta)
    offering_strategy.run_model()
    offering_strategy.calculate_results()

    # calculate oos profits
    offering_strategy.calculate_oos_profits()

    # Plot the in- and out-of-sample profits as histograms
    offering_strategy.plot_oos_profits()
    offering_strategy.plot_is_profits()

    # Step 1.5: Cross validation
    # Evaluate difference between expected in- and out-of-sample profits
    beta = 0.25
    plot_train_size_vs_profit_diff(beta=beta)

    # Perform 6-fold cross validation
    plot_train_size_vs_profit_diff_k_fold(beta=beta)