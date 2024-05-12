import gurobipy as gb
from gurobipy import GRB
import numpy as np
from scenario import DataInit
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

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
            train_size: int = 250,
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
        if self.price_scheme == 'two_price':
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
            BA_profits = gb.quicksum(
                self.pi[w] * (0.9 * self.imbalance_direction[t,w] + 1.2 * (1 - self.imbalance_direction[t,w])) *
                self.lambda_DA[t, w] * self.variables.Delta[t, w]
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
            BA_profits = UP_profits - DOWN_costs
        else:
            raise NotImplementedError
        if self.risk_type == 'averse':
            CVar = self.variables.zeta - 1/(1-self.alpha) * gb.quicksum(
                self.pi[w] * self.variables.eta[w] for w in self.SCENARIOS
            )
        else:
            CVar = 0

        objective = (1-self.beta) * (DA_profits + BA_profits) + self.beta * CVar
        self.model.setObjective(objective, gb.GRB.MAXIMIZE)

    def _build_constraints(self):
        self.constraints.imbalance_constraints = {t: {w: self.model.addLConstr(
            self.variables.Delta[t,w],
            gb.GRB.EQUAL,
            self.p_real[t,w] - self.variables.DA_dispatch[t],
            name='Imbalance constraint')
            for w in self.SCENARIOS} for t in self.TIMES}
        if self.price_scheme == 'two_price':
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
                          (0.9 * self.imbalance_direction[t,w] + 1.2 * (1 - self.imbalance_direction[t,w])) *
                          self.lambda_DA[t, w] * self.variables.Delta[t, w] for t in self.TIMES) +
                          self.variables.zeta - self.variables.eta[w],
                    gb.GRB.LESS_EQUAL,
                    0, name='eta constraint') for w in self.SCENARIOS}
            elif self.price_scheme == 'two_price':
                self.constraints.eta_constraint = {w: self.model.addLConstr(
                    - gb.quicksum(self.lambda_DA[t,w] * self.variables.DA_dispatch[t] +
                          0.9**(self.imbalance_direction[t,w]) * self.lambda_DA[t,w] * self.variables.Delta_UP[t,w] -
                          1.2**(1 - self.imbalance_direction[t,w]) * self.lambda_DA[t,w] * self.variables.Delta_DOWN[t,w]
                          for t in self.TIMES) + self.variables.zeta - self.variables.eta[w],
                    gb.GRB.LESS_EQUAL,
                    0, name='eta constraint') for w in self.SCENARIOS}
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

        if self.price_scheme == 'two_price':
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
                sum((0.9 * self.imbalance_direction[t, w] + 1.2 * (1 - self.imbalance_direction[t, w])) *
                    self.lambda_DA[t, w] * self.data.Delta_values[t][w] for t in self.TIMES
                )
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
            self.results.DA_profits[w] + self.results.BA_profits[w] for w in self.SCENARIOS
        }

        self.results.expected_profit = sum(
            self.pi[w] * (self.results.DA_profits[w] + self.results.BA_profits[w]) for w in self.SCENARIOS
        )
        if self.risk_type == 'averse':
            self.results.CVaR = self.data.zeta - 1/(1-self.alpha) * sum(
                self.pi[w] * self.data.eta_values[w] for w in self.SCENARIOS
            )

    def display_results(self):
        print()
        print("-------------------   RESULTS  -------------------")
        print("Expected profit: ")
        print(self.results.expected_profit)
        print("CVar: ")
        print(self.results.CVaR)

    def calculate_oos_profits(self):
        self.results.oos_DA_profits = []
        self.results.oos_BA_profits = []
        self.results.oos_profits = []

        for scenario in self.test_scenarios:
            wind = scenario['wind']
            lambda_DA = scenario['lambda']
            imbalance = scenario['system_balance']
            DA_profits = sum(
                lambda_DA[t] * self.data.DA_dispatch_values[t] for t in self.TIMES)
            if self.price_scheme == 'one_price':
                BA_profits = sum((0.9*imbalance[t] + 1.2*(1 - imbalance[t])) * lambda_DA[t] *
                                (wind[t] - self.data.DA_dispatch_values[t])
                                for t in self.TIMES)
            elif self.price_scheme == 'two_price':
                BA_profits = sum(
                    0.9 ** (imbalance[t]) * lambda_DA[t] * max(wind[t] - self.data.DA_dispatch_values[t], 0)
                    - 1.2 ** (1 - imbalance[t]) * lambda_DA[t] * max(self.data.DA_dispatch_values[t] - wind[t],0)
                    for t in self.TIMES)
            else:
                raise NotImplementedError
            self.results.oos_DA_profits.append(DA_profits)
            self.results.oos_BA_profits.append(BA_profits)
            total_profits = DA_profits + BA_profits
            self.results.oos_profits.append(total_profits)

        average_oos_profit = np.mean(self.results.oos_profits)
        print('BA profits in out-of-sample scenario 1: ', self.results.oos_BA_profits[0])
        print('Average out-of-sample profits: ', average_oos_profit)
        print('Expected in-sample profits: ', self.results.expected_profit)

    def plot_DA_dispatch(self, title: str):
        plt.figure()
        plt.grid(axis='y')
        plt.bar(self.data.DA_dispatch_values.keys(), self.data.DA_dispatch_values.values())
        plt.title(title)
        plt.xlabel('hour')
        plt.ylabel('power [MW]')
        plt.show()

    def plot_is_profits(self, title: str):
        plt.figure()
        in_sample_profits = list(self.results.total_profits.values())
        sns.histplot(data=in_sample_profits, kde=True, binwidth=10000)
        if self.risk_type == 'averse':
            plt.axvline(self.data.zeta, color='red', linestyle='--')
        #plt.hist(in_sample_profits, bins=30)#, density=True)
        plt.xlabel('In-sample profits [€]')
        plt.title(title)
        #plt.ylabel('Frequency')
        plt.show()

    def plot_oos_profits(self, title: str):
        plt.figure()
        #plt.hist(self.results.oos_profits, bins=30)#, density=True)
        sns.histplot(data=self.results.oos_profits, kde=True, binwidth=10000)
        plt.xlabel('Out-of-sample profits [€]')
        plt.title(title)
        #plt.ylabel('Frequency')
        plt.show()

    def plot_is_vs_oos_distribution(self):
        plt.figure()
        sns.kdeplot(data=list(self.results.total_profits.values()), label='In-sample')
        sns.kdeplot(data=self.results.oos_profits, color='skyblue', label='Out-of-sample')
        plt.xlim(0, 460000)
        plt.xlabel('Profits [€]')
        plt.ylabel('Density')
        plt.title("In- and Out-of-Sample Profit Distributions (" + self.price_scheme + ')')
        plt.legend()
        plt.show()


def plot_profits_comparison(one_price_os: OfferingStrategy, two_price_os: OfferingStrategy):
    profits = [
        np.mean(list(one_price_os.results.DA_profits.values())),
        np.mean(list(one_price_os.results.BA_profits.values())),
        one_price_os.results.expected_profit,
        np.mean(list(two_price_os.results.DA_profits.values())),
        np.mean(list(two_price_os.results.BA_profits.values())),
        two_price_os.results.expected_profit,
    ]
    df_profits = pd.DataFrame({
        'Profits [€]': profits,
        'Market': ['DA', 'BA', 'Total', 'DA', 'BA', 'Total'],
        'price scheme': ['one-price', 'one-price', 'one-price', 'two-price', 'two-price', 'two-price'],
    })
    plt.figure()
    plt.title('Comparison of profits for one-price and two-price')
    ax = sns.barplot(pd.DataFrame(df_profits), x="Market", y="Profits [€]", hue="price scheme", palette=['tab:blue','skyblue'])
    ax.bar_label(ax.containers[0], fontsize=10)
    ax.bar_label(ax.containers[1], fontsize=10)
    plt.show()

def plot_beta_vs_cvar(beta_values: list, price_scheme: str):
    expected_profits = []
    CVaRs = []
    offers = []
    for beta in beta_values:
        offering_strategy = OfferingStrategy(risk_type='averse', price_scheme=price_scheme, alpha=0.9, beta=beta)
        offering_strategy.run_model()
        offering_strategy.calculate_results()
        expected_profits.append(offering_strategy.results.expected_profit)
        CVaRs.append(offering_strategy.results.CVaR)
        offers.append(offering_strategy.data.DA_dispatch_values)

    # plot results with both lines and points
    plt.figure()
    plt.plot(CVaRs, expected_profits, label='Expected profit', marker='o', markersize=4)
    for ix, beta in enumerate(beta_values):
        if beta in [0, 0.05, 0.25, 1]:
            plt.annotate(round(float(beta), 2), (CVaRs[ix], expected_profits[ix]))
    plt.xlabel('CVaR [€]')
    plt.ylabel('Expected profit [€]')
    plt.title('CVaR vs. Expected Profit (' + price_scheme + ')')
    plt.show()

    # Plot the optimal DA offers for each beta
    indices = [0, 5, 10, 20]
    df = pd.DataFrame()
    for ix in indices:
        df[str(beta_values[ix])] = offers[ix]

    df.plot.bar(color=sns.dark_palette("#69d", reverse=True), figsize=(15, 4), width=0.8)
    plt.xlabel("Hour")
    plt.ylabel("Offered Power [MW]")
    plt.title(f"Day-ahead offering strategy ({price_scheme})")
    plt.legend(title="Beta", loc='upper center', bbox_to_anchor=(0.5, -0.2),
               fancybox=True, ncol=4)
    plt.tight_layout()
    plt.show()  # Plot the optimal bids for each beta

def profit_distribution_vs_beta(beta: np.ndarray, price_scheme: str):
    plt.figure()
    for b in beta:
        offstrat = OfferingStrategy(risk_type='averse', price_scheme=price_scheme, alpha=0.9, beta=b)
        offstrat.run_model()
        offstrat.calculate_results()
        sns.kdeplot(data=list(offstrat.results.total_profits.values()), label=str(b))
        #offstrat.calculate_oos_profits()
        #sns.kdeplot(data=offstrat.results.oos_profits,  label=str(b))
    plt.xlim(0, 460000)
    plt.xlabel('In-sample profits [€]')
    plt.title("In-Sample Profit Distributions for Different " + r'$\beta$' + "-values (" + price_scheme + ')')
    plt.legend(title="Beta values")
    plt.show()

def plot_train_size_vs_profit_diff(beta: float, price_scheme: str):
    train_sizes = np.linspace(100, 1100, 11).astype(int)
    profit_diffs = []

    for train_size in train_sizes:
        offering_strategy = OfferingStrategy(risk_type='averse', price_scheme=price_scheme, alpha=0.9, beta=beta, train_size=train_size)
        offering_strategy.run_model()
        offering_strategy.calculate_results()
        offering_strategy.calculate_oos_profits()

        avg_is_profits = np.mean(list(offering_strategy.results.total_profits.values()))
        avg_oos_profits = np.mean(offering_strategy.results.oos_profits)

        profit_diffs.append(abs(avg_is_profits - avg_oos_profits))

    plt.title("Train size vs profit differences")
    plt.plot(train_sizes, profit_diffs, marker='o')
    plt.grid(True)
    plt.xlabel("Train size")
    plt.ylabel("Absolute profit difference [€]")
    plt.show()

def plot_train_size_vs_profit_diff_k_fold(beta: float, price_scheme: str):
    train_sizes = [100, 200, 300, 400, 600]
    profit_diffs = []

    for train_size in train_sizes:
        avg_is_profits = []
        avg_oos_profits = []
        for k in range(1200 // train_size):
            offering_strategy = OfferingStrategy(risk_type='averse', price_scheme=price_scheme, alpha=0.9, beta=beta,
                                                 train_size=train_size, k=k)
            offering_strategy.run_model()
            offering_strategy.calculate_results()
            offering_strategy.calculate_oos_profits()

            avg_is_profits.append(np.mean(list(offering_strategy.results.total_profits.values())))
            avg_oos_profits.append(np.mean(offering_strategy.results.oos_profits))

        profit_diffs.append(np.mean(abs(np.array(avg_is_profits) - np.array(avg_oos_profits))))

    plt.title("Train size vs profit differences using k-fold cross validation")
    plt.plot(train_sizes, profit_diffs, marker='o')
    plt.grid(True)
    plt.xlabel("Train size")
    plt.ylabel("Absolute profit difference [€]")
    plt.show()


if __name__ == '__main__':
    """ Step 1.1: One-price """
    # Create and run optimization problem
    one_price_os = OfferingStrategy(risk_type='neutral', price_scheme='one_price')
    one_price_os.run_model()

    # Calculate results
    one_price_os.calculate_results()

    # Plot optimal day-ahead dispatch
    one_price_os.plot_DA_dispatch(title='Optimal Day-Ahead Dispatch (one-price)')

    # Plot the in-sample profits as a histogram
    one_price_os.plot_is_profits(title='In-Sample Profit Distribution (one-price)')

    """ Step 1.2: Two-price """
    # Create and run optimization problem
    two_price_os = OfferingStrategy(risk_type='neutral', price_scheme='two_price')
    two_price_os.run_model()

    # Calculate results
    two_price_os.calculate_results()

    # Plot optimal day-ahead dispatch
    two_price_os.plot_DA_dispatch(title='Optimal Day-Ahead Dispatch (two-price)')

    # Plot the in-sample profits as a histogram
    two_price_os.plot_is_profits(title='In-Sample Profit Distribution (two-price)')

    # Plot profit comparison
    plot_profits_comparison(one_price_os, two_price_os)

    """ Step 1.3: Risk analysis"""
    beta_values = np.linspace(0, 1, 21)

    # One-price
    plot_beta_vs_cvar(beta_values, 'one_price')

    # For one-price scheme the optimal beta is decided to be 0.25
    beta_one_price = 0.25
    one_price_os_risk = OfferingStrategy(risk_type='averse', price_scheme='one_price', alpha=0.9, beta=beta_one_price)
    one_price_os_risk.run_model()
    one_price_os_risk.calculate_results()
    one_price_os_risk.plot_DA_dispatch(title='Optimal Day-Ahead Dispatch (one-price)')

    # Two-price
    plot_beta_vs_cvar(beta_values, 'two_price')

    # For two-price scheme the optimal beta is decided to be 0.25
    beta_two_price = 0.25
    two_price_os_risk = OfferingStrategy(risk_type='averse', price_scheme='two_price', alpha=0.9, beta=beta_two_price)
    two_price_os_risk.run_model()
    two_price_os_risk.calculate_results()
    
    # Plot profit distribution for different beta values
    profit_distribution_vs_beta(np.linspace(0, 1, 5), 'one_price')
    profit_distribution_vs_beta(np.linspace(0, 1, 5), 'two_price')

    """ Step 1.4: Out-of-sample simulation """
    # calculate oos profits for one-price
    one_price_os_risk.calculate_oos_profits()
    one_price_os_risk.plot_oos_profits(title='Out-of-Sample Profit Distribution (one-price)')
    one_price_os_risk.plot_is_profits(title='In-Sample Profit Distribution (one-price)')
    one_price_os_risk.plot_is_vs_oos_distribution()

    # calculate oos profits for two-price
    two_price_os_risk.calculate_oos_profits()
    two_price_os_risk.plot_oos_profits(title='Out-of-Sample Profit Distribution (two-price)')
    two_price_os_risk.plot_is_profits(title='In-Sample Profit Distribution (two-price)')
    two_price_os_risk.plot_is_vs_oos_distribution()

    """ Step 1.5: Cross validation """
    # Evaluate difference between expected in- and out-of-sample profits
    plot_train_size_vs_profit_diff(beta=beta_one_price, price_scheme='one_price')
    plot_train_size_vs_profit_diff(beta=beta_two_price, price_scheme='two_price')

    # Perform 6-fold cross validation
    plot_train_size_vs_profit_diff_k_fold(beta=beta_one_price, price_scheme='one_price')
    plot_train_size_vs_profit_diff_k_fold(beta=beta_two_price, price_scheme='two_price')
