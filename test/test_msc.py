"""
    Unittest for the MSC.
"""

import unittest
import numpy as np
import os
import pickle
import shutil
from numpy.testing import assert_array_equal, assert_array_almost_equal
import pandas as pd
from sklearn.model_selection import train_test_split
from itertools import combinations
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from usecases.setcover.generate_instances import generate_msc_instances, generate_training_and_test_sets
from usecases.setcover.generate_instances import _linear_function, load_msc
from usecases.setcover.stochastic_algorithm import evaluate_stochastic_algo, StochasticAlgorithm
from usecases.setcover.solve_instances import MinSetCoverProblem
from usecases.setcover.predict_then_optimize import PoissonNeuralNetwork, negative_log_likelihood

########################################################################################################################


class MyTestCase(unittest.TestCase):

    # Set class variables
    num_prods = 200
    num_sets = 1000
    density = 0.02
    min_lmbd = 1
    max_lmbd = 10
    lmbd = np.random.randint(low=min_lmbd, high=max_lmbd, size=(num_prods,))
    num_scenarios = [1, 2, 4, 10, 20, 50, 75, 100, 250]
    num_instances = 500
    seed = 0
    data_path = os.path.join('data',
                             'msc',
                             f'{num_prods}x{num_sets}',
                             'linear',
                             f'{num_instances}-instances',
                             f'seed-{seed}')
    test_split = .5
    max_allowed_error = 0.1

    def test_instances_generation(self):
        """
        Check that instances differ only for the demands.
        :return:
        """

        num_instances = 100
        observables = np.random.randint(1, 10, size=num_instances)

        instances = \
            generate_msc_instances(num_instances=num_instances,
                                   num_sets=self.num_sets,
                                   num_products=self.num_prods,
                                   density=self.density,
                                   observables=observables)

        for pair in combinations(instances, 2):

            self.assertEqual(pair[0].num_sets, pair[0].num_sets)
            self.assertEqual(pair[0].num_products, pair[0].num_products)
            self.assertEqual(pair[0].density, pair[0].density)
            assert_array_equal(pair[0].availability, pair[1].availability)
            assert pair[0].demands.shape == pair[1].demands.shape
            assert not np.array_equal(pair[0].demands, pair[1].demands)
            assert_array_equal(pair[0].set_costs, pair[1].set_costs)
            assert_array_equal(pair[0].prod_costs, pair[1].prod_costs)

    def test_stochastic_algo_eval(self):
        """
        Compute the solution of the stochastic algorithm with a number od scenarios equal to the instances size.
        The objective value of the Gurobi solver must be the same of the one computed by the evaluation function.
        :return:
        """

        # Set the random seed to ensure reproducibility
        np.random.seed(self.seed)

        observables = np.random.randint(1, 10, size=self.num_instances)

        assert max(self.num_scenarios) <= self.num_instances, \
            "The maximum number of scenarios can not br grater than the size of the sampled instances"

        instances = \
            generate_msc_instances(num_instances=self.num_instances,
                                   num_sets=self.num_sets,
                                   num_products=self.num_prods,
                                   density=self.density,
                                   observables=observables)

        print()

        optimal_costs = list()
        for idx, inst in enumerate(instances):
            problem = MinSetCoverProblem(instance=inst)
            optimal_sol, optimal_cost = problem.solve()
            optimal_costs.append(optimal_cost)
            print(f'Solved {idx+1}/{len(instances)} instances')

        all_mean_stochastic_algo_costs = list()

        for num_scenarios in self.num_scenarios:

            # Create the stochastic algorithm instance and solve the problem
            instance = StochasticAlgorithm(instances=instances, num_scenarios=num_scenarios)
            res = instance.solve()

            # Evaluate the results
            res = \
                evaluate_stochastic_algo(res=res,
                                         num_scenarios=num_scenarios,
                                         instances=instances,
                                         optimal_costs=optimal_costs)

            all_mean_stochastic_algo_costs.append(res['Stochastic algo mean cost'])

            print()
            print('-' * 100)
            print()

        # Sanity checks
        # FIXME: the result of the argmin operator may a sequence object
        best_cost_idx = np.argmin(all_mean_stochastic_algo_costs)
        best_cost_value = all_mean_stochastic_algo_costs[best_cost_idx]
        self.assertEqual(best_cost_idx, len(all_mean_stochastic_algo_costs) - 1)
        self.assertGreaterEqual(best_cost_value, res['Mean optimal cost'])

    def test_probabilistic_regressor_training(self):
        """
        Check whether the training of the probabilistic regressor model is successful.
        :return:
        """
        # Set the random seed to ensure reproducibility
        np.random.seed(self.seed)

        # Load the instances
        instances = list()

        print('Loading instances...')
        for f in os.listdir(self.data_path):
            if os.path.isdir(os.path.join(self.data_path, f)):
                instances.append(load_msc(os.path.join(self.data_path, f, 'instance.pkl')))
        print('Finished')

        # Split between training and test instances
        train_instances, test_instances = \
            train_test_split(instances, test_size=self.test_split, random_state=self.seed)
        train_obs = np.asarray([inst.observables for inst in train_instances])
        test_obs = np.asarray([inst.observables for inst in test_instances])
        train_obs = np.expand_dims(train_obs, axis=1)
        test_obs = np.expand_dims(test_obs, axis=1)
        train_lambdas = np.asarray([inst.lmbds for inst in train_instances])
        test_lambdas = np.asarray([inst.lmbds for inst in test_instances])

        # Create and train the probabilistic regressor model
        model = PoissonNeuralNetwork(input_shape=(1,), output_shape=self.num_prods)
        model.compile(optimizer=Adam(), loss=negative_log_likelihood)
        history = \
            model.fit(x=train_obs,
                      y=train_lambdas,
                      validation_split=0.2,
                      epochs=2000,
                      batch_size=8,
                      callbacks=[EarlyStopping(monitor='val_loss',
                                               patience=20,
                                               restore_best_weights=True)])

        # Evaluate the probabilistic regressor
        preds = model(test_obs)
        lmbds_hat = preds.rate.numpy()
        maes = list()
        for i in range(self.num_prods):
            mae = np.mean(np.abs(lmbds_hat[:, i] - test_lambdas[:, i]))
            maes.append(mae)

        worst_error = np.max(maes)
        self.assertLess(worst_error, self.max_allowed_error)


########################################################################################################################


