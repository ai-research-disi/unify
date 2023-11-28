"""
    Utility methods for the Stochasti algorithm.
"""

import pickle
import argparse
import numpy as np
from gurobipy import Model, GRB, MVar, Env
from sklearn.model_selection import train_test_split
import os
import shutil
from usecases.setcover.generate_instances import load_msc, MinSetCover, StochasticMinSetCover
from typing import Union, List, Tuple

########################################################################################################################


class StochasticAlgorithm:
    """
    Implementation of Sample Average Approximation algorithm to solve the stochastic version of the MSC.
    """
    def __init__(self,
                 instances: List[MinSetCover],
                 num_scenarios: int,
                 output_flag: int = 0):

        assert len(instances) >= num_scenarios, "The number of scenarios must be less or equal the number of instances"

        self._instances = instances
        self._num_scenarios = num_scenarios

        set_costs = self._instances[0].set_costs
        prod_costs = self._instances[0].prod_costs
        availability = self._instances[0].availability
        num_sets = self._instances[0].num_sets
        num_products = self._instances[0].num_products

        env = Env(empty=True)
        env.setParam("OutputFlag", output_flag)
        env.start()
        # Create the Gurobi model
        self._problem = Model(env=env)

        self._problem.setParam('OutputFlag', output_flag)

        # Save the scenarios demands
        self._scenario_demands = list()

        # Penalty cost due to the violation of the constraints
        self._set_penalty(self._instances[0])
        penalty_cost = 0

        # self._instance.num_sets is J
        J = num_sets

        # This is the set of decision variables
        X = [self._problem.addVar(vtype=GRB.INTEGER, lb=0, name=f'x_{j}') for j in range(J)]

        # self._instance.num_products is I
        I = num_products

        X = np.asarray(X)

        cover_constraint = availability @ X

        # Here is the code for the first constraint of the problem (second line of the formulation)
        # omega is \omega and self._num_scenarios is \Omega
        for omega in range(self._num_scenarios):
            print(f'[Stochastic algorithm] - Scenario: {omega}')

            # This is d_{0,...I, \omega}
            # This is a matrix of shape (I, )
            # self._instance.new()
            d = self._instances[omega].demands
            self._scenario_demands.append(d)

            # This is the set of indicator variables
            Z = list()
            # This is the set of slack variables
            S = list()

            # Add the indicator constraints
            for i in range(0, I):

                indicator_var = self._problem.addVar(vtype=GRB.BINARY, name=f'z_({i},{omega})')
                slack_var = self._problem.addVar(vtype=GRB.CONTINUOUS, name=f's_({i}, {omega})')

                # Add the indicator and slack variables
                Z.append(indicator_var)
                S.append(slack_var)

                # RHS of the indicator constraint
                lhs_constraint = slack_var + cover_constraint[i]

                # Indicator constraint
                self._problem.addGenConstrIndicator(binvar=indicator_var,
                                                    binval=True,
                                                    lhs=lhs_constraint,
                                                    sense=GRB.GREATER_EQUAL,
                                                    rhs=d[i],
                                                    name=f'Indicator_constraint_({omega},{i})')

                # Add demands satisfaction constraint
                self._problem.addConstr(cover_constraint[i] >= d[i] * (1 - indicator_var))

            # Add the penalty when the indicator constraint is violated
            S_as_matrix = MVar(S)
            penalty_cost += self._penalty @ S_as_matrix

        # Convert the list of decision variables in matrix form
        X_as_matrix = MVar(X)

        # Objective function
        obj = set_costs @ X_as_matrix + penalty_cost / self._num_scenarios
        self._problem.setObjective(obj, GRB.MINIMIZE)

        self._problem.write('model.lp')

        self._scenario_demands = np.asarray(self._scenario_demands)

    def _set_penalty(self,
                     instance: StochasticMinSetCover):
        """
        The penalty is the same of the Set Cover problem.
        :param instance: StochasticMinSetCover; the MSC instance
        :return:
        """

        self._penalty = instance.prod_costs

    def solve(self) -> Tuple[np.ndarray, float]:
        """
        Solve the optimization problem.
        :return: numpy.array, float; solution and its objective value.
        """

        # Save the model
        print('[Stochastic algorithm] Starting optimization...')
        self._problem.optimize()
        runtime = self._problem.Runtime
        print(f'[Stochastic algorithm] Optimization finished - Elapsed: {runtime}')
        status = self._problem.status

        assert status == GRB.Status.OPTIMAL, "Solution is not optimal"

        solution = self._problem.getVars()
        obj_val = self._problem.objVal

        decision_vars = list()
        indicator_vars = list()
        for sol in solution:
            if sol.VarName.startswith('x'):
                # print_str += f'{sol.VarName}: {sol.X} - '
                decision_vars.append(sol.X)
            elif sol.VarName.startswith('z'):
                indicator_vars.append(sol.X)
        # print_str += f'\nSolution cost: {obj_val}'

        decision_vars = np.asarray(decision_vars)
        indicator_vars = np.asarray(indicator_vars)

        # Save results in a dictionary
        res = dict()
        res['Solution'] = decision_vars
        res['Indicator variables'] = indicator_vars
        res['Costs'] = obj_val
        res['Runtime'] = runtime

        return res

    @property
    def instance(self):
        return self._instance

    @property
    def num_scenarios(self):
        return self._num_scenarios

    @property
    def penalty(self):
        return self._penalty

    def __str__(self):
        return str(self._instance)

########################################################################################################################


def evaluate_stochastic_algo(res: Union[str, dict],
                             instances: List[MinSetCover],
                             optimal_costs: List[float],
                             num_scenarios: int):
    """
    Evaluate the stochastic algorithm on a set of instances.
    :param res: string or dict; either the path where the solution found by the stochastic algorithm is loaded from
                                or the solutions itself as dictionary.
    :param instances: list of usecases.setcover.MinSetCover; the list of MSC instances.
    :param optimal_costs: list of float; optimal cost for each instance.
    :param num_scenarios: int; the number of scenarios used for the stochastic algorithm approach.
    :return:
    """

    # If 'res' is a string then load the solution from the 'res' directory...
    if isinstance(res, str):
        # Results path
        path = os.path.join(res, f'{num_scenarios}-scenarios')
        assert os.path.exists(path), f"{path} directory does not exist"

        # Load the results from pickle
        res_file = open(os.path.join(res, f'{num_scenarios}-scenarios', 'res.pkl'), 'rb')
        res = pickle.load(res_file)
        res_file.close()

    assert isinstance(res, dict), "res must be a dictionary"
    assert 'Solution' in res.keys(), '"Solution" key is expected in the results pickle'
    stochastic_algo_sol = res['Solution']

    # Get the elements production
    production = instances[0].availability @ stochastic_algo_sol
    # The solution is unique so we use it to evaluate the whole set of instances
    production = np.tile(production, (len(instances), 1))
    # Compute the not satisfied demands
    demands = [instance.demands for instance in instances]
    demands = np.asarray(demands)
    not_satisfied_demands = demands - production

    # Compute cost for using the sets
    real_cost = instances[0].set_costs @ stochastic_algo_sol

    # Compute the cost for not satisfied demands
    not_satisfied_demands = np.clip(not_satisfied_demands, a_min=0, a_max=None)
    not_satisfied_demands_cost = not_satisfied_demands @ instances[0].prod_costs
    cost = not_satisfied_demands_cost + real_cost

    stochastic_algo_mean_cost = np.mean(cost)
    stochastic_algo_std_cost = np.std(cost)

    mean_optimal_cost = np.mean(optimal_costs)
    std_optimal_cost = np.std(optimal_costs)

    print_str = f'[Stochastic algorithm] Num. scenarios: {num_scenarios} | Mean optimal cost: {mean_optimal_cost} | ' + \
                f'Stochastic algorithm mean cost: {stochastic_algo_mean_cost}'
    print(print_str + '\n')
    print('-'*len(print_str) + '\n')

    res['Mean optimal cost'] = mean_optimal_cost
    res['Std optimal cost'] = std_optimal_cost
    res['Stochastic algo mean cost'] = stochastic_algo_mean_cost
    res['Stochastic algo std cost'] = stochastic_algo_std_cost

    return res

########################################################################################################################


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("datadir", type=str, help="Data directory")
    parser.add_argument("resdir", type=str, help="Results directory")
    parser.add_argument("--num-prods", type=int, help="Number of products")
    parser.add_argument("--num-sets", type=int, help="Number of sets")
    parser.add_argument("--num-instances", type=int, help="Number of generated instances")
    parser.add_argument("--evaluated-instances", type=int, help="Number of test set instances used for the evaluation")
    parser.add_argument("--test-split", type=float, help="Fraction of the instances used as test")
    parser.add_argument("--seed", type=int, help="Seed to ensure reproducibility of the results")
    parser.add_argument("--num-scenarios", type=int, nargs='+', help="Number of scenarios for the stochastic algo")

    args = parser.parse_args()

    # Set some constant values
    NUM_PRODS = int(args.num_prods)
    NUM_SETS = int(args.num_sets)
    NUM_INSTANCES = int(args.num_instances)
    NUM_EVALUATED_INSTANCES = int(args.evaluated_instances)
    TEST_SPLIT = float(args.test_split)
    SEED = int(args.seed)
    NUM_SCENARIOS = args.num_scenarios
    DATA_PATH = args.datadir
    DATA_PATH = os.path.join(DATA_PATH,
                             f'{NUM_PRODS}x{NUM_SETS}',
                             'linear',
                             f'{NUM_INSTANCES}-instances',
                             f'seed-{SEED}')
    RESULTS_DIR = args.resdir
    LOG_DIR = os.path.join(RESULTS_DIR,
                           'stochastic-algo',
                           'mean',
                           f'{NUM_PRODS}x{NUM_SETS}',
                           'linear',
                           f'{NUM_INSTANCES}-instances',
                           f'seed-{SEED}')

    # The maximum number of allowed scenarios is training set size
    assert max(NUM_SCENARIOS) <= int(NUM_INSTANCES * (1 - TEST_SPLIT)), \
        "The maximum number of allowed scenarios is training set size"

    # Set the random seed to ensure reproducibility
    np.random.seed(SEED)

    # Load the instances
    instances = list()
    optimal_costs = list()

    print('Loading instances...')
    for f in os.listdir(DATA_PATH):
        if os.path.isdir(os.path.join(DATA_PATH, f)):
            instances.append(load_msc(os.path.join(DATA_PATH, f, 'instance.pkl')))
            optimal_cost = pickle.load(open(os.path.join(DATA_PATH, f, 'optimal-cost.pkl'), 'rb'))
            optimal_costs.append(optimal_cost)
    print('Finished')

    # Split between training and test instances
    train_instances, test_instances, \
        train_optimal_costs, test_optimal_costs = \
            train_test_split(instances, optimal_costs, test_size=TEST_SPLIT, random_state=SEED)

    all_res = list()

    for num_scenarios in NUM_SCENARIOS:
        log_dir = os.path.join(LOG_DIR,
                               f'{num_scenarios}-scenarios')

        # Remove the logging folder
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        else:
            shutil.rmtree(log_dir)
            os.makedirs(log_dir)

        # Create the stochastic algorithm instance and solve the problem
        instance = StochasticAlgorithm(instances=train_instances, num_scenarios=num_scenarios)
        res = instance.solve()
        pickle.dump(res, open(os.path.join(log_dir, 'res.pkl'), 'wb'))

        # Evaluate the results on training set
        print('\nTRAINING SET')
        evaluate_stochastic_algo(res=LOG_DIR,
                                 num_scenarios=num_scenarios,
                                 instances=train_instances,
                                 optimal_costs=train_optimal_costs)

        # Evaluate the results on test set
        print('TEST SET')
        evaluate_stochastic_algo(res=LOG_DIR,
                                 num_scenarios=num_scenarios,
                                 instances=test_instances[:NUM_EVALUATED_INSTANCES],
                                 optimal_costs=test_optimal_costs)

        print()
