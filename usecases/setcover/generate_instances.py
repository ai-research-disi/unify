"""
    Set of methods to generate the deterministic and stochastic MSC.
"""

import os
import pickle
import numpy as np
from tabulate import tabulate
import gym
from gym.spaces import Box
import random
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import copy
import inspect
import shutil
import argparse
from usecases.setcover.solve_instances import MinSetCoverProblem, compute_cost
from typing import List, Callable, Tuple

########################################################################################################################

MSC_ATTRIBUTES = {'availability', 'demands', 'num_products', 'num_sets', 'prod_costs', 'set_costs'}

########################################################################################################################


def _linear_function(a):
    return lambda x: a * x

########################################################################################################################


class MinSetCover:
    """
    Minimum Set Cover class.

    Attributes:
        num_sets: int; number of sets.
        num_products: int; number of products.
        set_costs: np.array of floats; cost for each set.
        prod_costs: np.array of floats; cost for each product.
        availability; np.array of int; availability of each product in a set.
        demands: np.array of float; demand for each product.
    """

    def __init__(self,
                 num_sets: int,
                 num_products: int,
                 density: float,
                 availability: np.ndarray = None,
                 demands: np.ndarray = None,
                 set_costs: np.ndarray = None,
                 prod_costs: np.ndarray = None,
                 observables: np.ndarray = None,
                 lmbds: np.ndarray = None):

        self._num_sets = num_sets
        self._num_products = num_products
        self._density = density
        self._observables = observables
        self._lmbds = lmbds

        # You can choose to give some MSC parameters as input or create them from scratch

        if set_costs is None:
            # Uniform random generation of the costs in the interval [1, 100]
            self._set_costs = np.random.randint(low=1, high=100, size=num_sets)
        else:
            self._set_costs = set_costs

        if demands is not None:
            self._demands = demands

        if availability is None:
            self._availability = self._generate_availability()
        else:
            self._availability = availability

        if prod_costs is None:
            self._set_prod_cost()
        else:
            self._prod_costs = prod_costs

    def _generate_availability(self) -> np.ndarray:
        """
        Generate the availability of each product in each set.
        :return: numpy.array of shape (num_products, num_sets); 0-1 matrix for each product-set pair.
        """

        assert 0 < self._density < 1, "Density must be in ]0,1["
        assert self._num_products is not None, "_num_products must be initialized"
        assert self._num_sets is not None, "_num_sets must be initialized"

        availability = np.zeros(shape=(self._num_products, self._num_sets), dtype=np.int8)

        for row in range(self._num_products):
            first_col = -1
            second_col = -1
            while first_col == second_col:
                first_col = np.random.randint(low=0, high=self._num_sets, size=1)
                second_col = np.random.randint(low=0, high=self._num_sets, size=1)
                availability[row, first_col] = 1
                availability[row, second_col] = 1

        for col in range(self._num_sets):
            row = np.random.randint(low=0, high=self._num_products, size=1)
            availability[row, col] = 1

        # Check that all the products are available in at least two sets
        available_products = np.sum(availability, axis=1) > 1

        # Check that all the sets have at least one product
        at_least_a_prod = np.sum(availability, axis=0) > 0

        density = np.clip(self._density - np.mean(availability), a_min=0, a_max=None)
        availability += np.random.choice([0, 1], size=(self._num_products, self._num_sets), p=[1 - density, density])
        availability = np.clip(availability, a_min=0, a_max=1)

        print(f'[MinSetCover] - True density: {np.mean(availability)}')

        assert available_products.all(), "Not all the products are available"
        assert at_least_a_prod.all(), "Not all set cover at least a product"

        return availability

    def _set_prod_cost(self):
        """
        The product costs are set according to the min cost of among the sets that cover and multiply it by 10.
        :return:
        """

        assert self._set_costs is not None, "set_costs must be initialized"

        self._prod_costs = np.zeros(shape=(self.num_products, ))
        for idx in range(self._num_products):
            prod_availability = self._availability[idx]
            prod_costs = prod_availability * self._set_costs
            prod_costs = prod_costs[np.nonzero(prod_costs)]
            max_cost = np.max(prod_costs)
            self._prod_costs[idx] = max_cost * 10

    @property
    def num_sets(self):
        return self._num_sets

    @num_sets.setter
    def num_sets(self, value):
        self._num_sets = value

    @property
    def num_products(self):
        return self._num_products

    @num_products.setter
    def num_products(self, value):
        self._num_products = value

    @property
    def density(self):
        return self._density

    @density.setter
    def density(self, value):
        self._density = value

    @property
    def set_costs(self):
        return self._set_costs

    @set_costs.setter
    def set_costs(self, value):
        self._set_costs = value

    @property
    def prod_costs(self):
        return self._prod_costs

    @prod_costs.setter
    def prod_costs(self, value):
        self._prod_costs = value

    @property
    def availability(self):
        return self._availability

    @availability.setter
    def availability(self, value):
        self._availability = value

    @property
    def demands(self):
        return self._demands

    @demands.setter
    def demands(self, value):
        self._demands = value

    @property
    def observables(self):
        return self._observables

    @observables.setter
    def observables(self, value):
        self._observables = value

    @property
    def lmbds(self):
        return self._lmbds

    @lmbds.setter
    def lmbds(self, value):
        self._lmbds = value

    def new(self):
        pass

    def dump(self, filepath: str):
        """
        Save the MSC instance in a pickle.
        :param filepath: str; where the instance is saved to.
        :return:
        """
        msc_dict = dict()

        # We save all the readable properties
        for member_name, member_value in inspect.getmembers(self):
            if not member_name.startswith('_') and not inspect.ismethod(member_value):
                msc_dict[member_name] = member_value

        pickle.dump(msc_dict, open(filepath, 'wb'))

    def __str__(self):
        print_str = ""
        print_str += f"Num. of sets: {self._num_sets} | Num. of products: {self._num_products}\n"
        header = [f"Cost for set n.{idx}" for idx in range(self._num_sets)]
        print_str += tabulate(np.expand_dims(self._set_costs, axis=0), headers=header, tablefmt='pretty') + '\n'
        header = [f"Cost for product n.{idx}" for idx in range(self._num_products)]
        print_str += tabulate(np.expand_dims(self._prod_costs, axis=0), headers=header, tablefmt='pretty') + '\n'
        header = [f"Demand for product n.{idx}" for idx in range(self._num_products)]
        print_str += tabulate(np.expand_dims(self._demands, axis=0), headers=header, tablefmt='pretty') + '\n'
        header = [f'Availability for set n.{idx}' for idx in range(0, self._num_sets)]
        availability = list()
        for prod_idx in range(0, self._num_products):
            availability.append([f'Product n. {prod_idx}'] + list(self._availability[prod_idx, :]))
        print_str += tabulate(availability, headers=header, tablefmt='pretty') + '\n'

        return print_str

########################################################################################################################


class StochasticMinSetCover(MinSetCover):
    """
    Minimum Set Covering with stochastic demands.
    """
    def __init__(self,
                 num_sets: int,
                 num_products: int,
                 density: float,
                 observable_lambda_fun: Callable,
                 availability: np.ndarray = None,
                 demands: np.ndarray = None,
                 set_costs: np.ndarray = None,
                 prod_costs: np.ndarray = None):

        super().__init__(num_sets=num_sets,
                         num_products=num_products,
                         density=density,
                         availability=availability,
                         demands=demands,
                         set_costs=set_costs,
                         prod_costs=prod_costs)

        self._observable_lambda_fun = observable_lambda_fun
        self._observables = list()

    def new(self, observable, idx):
        """
        Generate a new set of demands sampling from a Poisson distribution.
        :return:
        """
        lmbd = list()

        # The Poisson rates are created according to the specified relationship
        for f in self._observable_lambda_fun:
            lmbd.append(f(observable))

        assert len(lmbd) == self._num_products, "Lambda must have size equals to the number of products"

        self._demands = np.random.poisson(lmbd, size=self._num_products)
        self.observables = observable
        self.lmbds = lmbd

########################################################################################################################


class MinSetCoverEnv(gym.Env):
    """
    Gym wrapper for MSC.

    Attributes:
        current_instance: MinSetCover; current MSC instance.
        max_episode_length: int; max episode length.
        train_instances: list of MinSetCover; the training instances.
        test_instances; list of MinSetCover; the test instances.
        demands_scaler; sklearn.preprocessing.StandardScaler; scaler used to preprocess the demands.
    """

    # This is a gym.Env variable that is required by garage for rendering
    metadata = {
        "render.modes": ["ascii"]
    }

    def __init__(self,
                 instances_filepath: str,
                 num_prods: int,
                 num_sets: int,
                 seed: int,
                 test_split: float = 0.5):

        super(MinSetCoverEnv, self).__init__()

        self._num_prods = num_prods
        self._num_sets = num_sets
        self._instances_filepath = instances_filepath

        # Set the action and observation spaces required by Gym
        self.action_space = Box(low=0, high=np.inf, shape=(self._num_prods, ), dtype=np.float32)
        self.observation_space = Box(low=0, high=np.inf, shape=(1, ), dtype=np.float32)

        # Load instances from file
        print('[MinSetCoverEnv] - Loading instances...')
        instances, demands, self._optimal_costs = self._load_instances()
        print('[MinSetCoverEnv] - Finished')

        # Split between training and test sets
        self._train_instances, self._test_instances, \
            train_demands, test_demands = \
                train_test_split(instances, demands, test_size=test_split, random_state=seed)

        # Randomly select one of the instances
        self._current_instance = random.sample(self._train_instances, k=1)[0]

        # Standardize demands
        self._demands_scaler = StandardScaler()
        self._demands_scaler.fit_transform(train_demands)

    def _load_instances(self) -> Tuple[List[MinSetCover], np.ndarray, List[float]]:
        """
        Load instances from file.
        :return: list of MinSetCover, list of numpy.array; the generated instances and corresponding demands.
        """
        instances = list()
        demands = list()
        optimal_costs = list()

        for f in os.listdir(self._instances_filepath):
            path = os.path.join(self._instances_filepath, f)

            if os.path.isdir(path):
                instance_path = os.path.join(path, 'instance.pkl')
                optimal_cost_path = os.path.join(path, 'optimal-cost.pkl')
                assert os.path.exists(instance_path), "instance.pkl not found"
                assert os.path.exists(optimal_cost_path), "optimal-cost.pkl not found"

                instance = load_msc(instance_path)
                instances.append(instance)
                demands.append(instance.demands)
                cost = pickle.load(open(optimal_cost_path, 'rb'))
                optimal_costs.append(cost)

        return instances, demands, optimal_costs

    @property
    def current_instance(self):
        return self._current_instance

    @current_instance.setter
    def current_instance(self, val):
        self._current_instance = val

    @property
    def max_episode_length(self):
        return 1

    @property
    def demands_scaler(self):
        return self._demands_scaler

    @property
    def train_instances(self):
        return self._train_instances

    @property
    def test_instances(self):
        return self._test_instances

    def reset(self) -> np.ndarray:
        """
        Reset the environment randomly selecting one of the instances.
        :return: numpy.array; the observations.
        """
        self._current_instance = random.sample(self._train_instances, k=1)[0]
        observables = self._current_instance.observables

        # FIXME: this is useless for ndarray
        observables = np.array([observables])

        return observables

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """
        What this method does:
        - Scale the demands.
        - Round demands to the closest integer.
        - Solve a copy of the true MSC replacing predicted demands.
        - Compute the real cost of the obtained solution.
        - Compute the penalty if demands are not satisfied.
        :param action: numpy.array; the action.
        :return: numpy.array, float, boolean, dict; observations, reward, end of episode flag, info.
        """
        action = np.expand_dims(action, axis=0)
        action = self._demands_scaler.inverse_transform(action)
        action = np.squeeze(action)
        action = np.rint(action)
        action = np.clip(action, a_min=self.action_space.low, a_max=self.action_space.high)
        virtual_msc = copy.deepcopy(self._current_instance)
        virtual_msc.demands = action

        problem = MinSetCoverProblem(instance=virtual_msc)

        solution, _ = problem.solve()
        not_satisfied_demands = self._current_instance.demands - action

        # Compute true cost and optimality gap
        cost = compute_cost(instance=self._current_instance,
                            decision_vars=solution,
                            not_satisfied_demands=not_satisfied_demands)

        # Environment information
        info = {'Demands': action,
                'Solution': solution,
                'Cost': cost,
                'Solution': solution,
                'Action': action}

        observables = self._current_instance.observables

        # FIXME: this is useless for ndarray
        observables = np.array([observables])

        return observables, -cost, True, info

    def render(self, mode: str = 'human'):
        """
        Visualize the environment.
        :param mode: str; visualization mode.
        :return:
        """

        assert mode == 'human', "Only 'human' mode is supported"

        print(self.current_instance)
        problem = MinSetCoverProblem(self.current_instance, output_flag=0)
        solution, cost = problem.solve()

        print_str = f'Real optimal solution: {solution} | Real optimal cost: {cost}'
        print(print_str)

    def close(self):
        pass

########################################################################################################################


def generate_msc_instances(num_instances: int,
                           num_sets: int,
                           num_products: int,
                           density: float,
                           observables: np.ndarray) -> List[MinSetCover]:
    """
    Generate a random set of MSC instances.
    :param num_instances: int; the number of instances to be generated.
    :param num_sets: int; number of sets of the MSC.
    :param num_products: int; number of products (elements) of the MSC.
    :param density: float; the density of the availability matrix.
    :param observables: np.array; the observables associated to the lambda value of the Poisson distribution.
    :return:
    """

    assert len(observables) == num_instances, "The lenght of the observables must be equal to the number of instances"

    instances = list()

    fun = list()
    for i in range(num_products):
        f = _linear_function(np.random.randint(1, 5))
        fun.append(f)

    # Use a stochastic MSC instance as factory
    factory = StochasticMinSetCover(num_sets=num_sets,
                                    num_products=num_products,
                                    density=density,
                                    availability=None,
                                    demands=None,
                                    set_costs=None,
                                    prod_costs=None,
                                    observable_lambda_fun=fun)

    for idx in range(num_instances):
        factory.new(observables[idx], idx=idx)
        inst = copy.deepcopy(factory)
        instances.append(inst)

    return instances

########################################################################################################################


def generate_training_and_test_sets(data_path: str,
                                    num_instances: int,
                                    num_sets: int,
                                    num_prods: int,
                                    density: float,
                                    min_lmbd: float,
                                    max_lmbd: float):
    """
    Generate and save on a file training and test MSC instances.
    :param data_path: string; where instances are saved to.
    :param num_instances: int; number of instances to be generated.
    :param num_sets: int; number of sets of the MSC.
    :param num_prods: int; number of products of the MSC.
    :param density: float; density of the availability matrix.
    :param min_lmbd: float; min value allowed for lambda.
    :param max_lmbd:float; max value allowed for lambda.
    :return:
    """

    # Generate the observables and the MSC instances
    observables = np.random.uniform(min_lmbd, max_lmbd, size=num_instances)
    instances = \
        generate_msc_instances(num_instances=num_instances,
                               num_sets=num_sets,
                               num_products=num_prods,
                               density=density,
                               observables=observables)

    # Create the data folder
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    else:
        shutil.rmtree(data_path)
        os.makedirs(data_path)

    # Save the generated instances
    for idx, inst in enumerate(instances):
        print(f'Saving instance {idx + 1}/{len(instances)}')

        problem = MinSetCoverProblem(instance=inst)
        optimal_sol, optimal_cost = problem.solve()

        instance_path = os.path.join(data_path, f'instance-{idx}')
        if not os.path.exists(instance_path):
            os.makedirs(instance_path)
        else:
            shutil.rmtree(instance_path)
            os.makedirs(instance_path)

        inst.dump(os.path.join(instance_path, f'instance.pkl'))
        pickle.dump(optimal_sol,
                    open(os.path.join(instance_path, f'optimal-sol.pkl'), 'wb'))
        pickle.dump(optimal_cost,
                    open(os.path.join(instance_path, f'optimal-cost.pkl'), 'wb'))


########################################################################################################################


def load_msc(filepath: str):
    """
    For the sake of simplicity, we saved the only MSC attributes.
    :param filepath: str; where the MSC instances are loaded from.
    :return:
    """
    attributes = pickle.load(open(filepath, 'rb'))
    msc = MinSetCover(**attributes)
    return msc

########################################################################################################################


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("datadir", type=str, help="Data directory")
    parser.add_argument("--min-lambda", type=int, help="Minimum value of lambda that can be generated")
    parser.add_argument("--max-lambda", type=int, help="Maximum value of lambda that can be generated")
    parser.add_argument("--num-prods", type=int, help="Number of products")
    parser.add_argument("--num-sets", type=int, help="Number of sets")
    parser.add_argument("--density", type=float, help="Density of the availability matrix")
    parser.add_argument("--num-instances", type=int, help="Number of generated instances")
    parser.add_argument("--seed", type=int, help="Seed to ensure reproducibility of the results")

    args = parser.parse_args()

    MIN_LMBD = int(args.min_lambda)
    MAX_LMBD = int(args.max_lambda)
    NUM_PRODS = int(args.num_prods)
    NUM_SETS = int(args.num_sets)
    DENSITY = float(args.density)
    NUM_INSTANCES = int(args.num_instances)
    SEED = int(args.seed)
    DATA_PATH = args.datadir
    DATA_PATH = os.path.join(DATA_PATH,
                             f'{NUM_PRODS}x{NUM_SETS}',
                             'linear',
                             f'{NUM_INSTANCES}-instances',
                             f'seed-{SEED}')

    # Set the random seed to ensure reproducibility
    np.random.seed(SEED)

    # Generate training and test set in the specified directory
    generate_training_and_test_sets(data_path=DATA_PATH,
                                    num_instances=NUM_INSTANCES,
                                    num_sets=NUM_SETS,
                                    num_prods=NUM_PRODS,
                                    density=DENSITY,
                                    min_lmbd=MIN_LMBD,
                                    max_lmbd=MAX_LMBD)

