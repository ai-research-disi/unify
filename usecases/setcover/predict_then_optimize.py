"""
    Utility function for the predict-then-optimize approach.
"""

import numpy as np
import os
import pickle
import argparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow_probability as tfp
import seaborn as sns
import copy
import shutil
from usecases.setcover.generate_instances import load_msc
from usecases.setcover.stochastic_algorithm import StochasticAlgorithm, evaluate_stochastic_algo
sns.set_style('darkgrid')

########################################################################################################################


class PoissonNeuralNetwork(Model):
    def __init__(self,
                 input_shape: int,
                 output_shape: int):
        super(PoissonNeuralNetwork, self).__init__()
        self._model = Sequential()
        self._model.add(Input(input_shape))
        self._model.add(Dense(units=output_shape, activation=tf.nn.softplus))

    def call(self, inputs):
        output = self._model(inputs)
        poisson = tfp.distributions.Poisson(rate=output)
        return poisson

    def save(self, savepath: str):
        self._model.save(savepath)

    def load(self, loadpath: str):
        self._model = tf.keras.models.load_model(loadpath)

########################################################################################################################


def negative_log_likelihood(y_true: np.ndarray,
                            y_hat: tfp.distributions.Distribution) -> np.ndarray:
    """
    Negative log-likelihood function.
    :param y_true: numpy.array; the ground truth.
    :param y_hat: tfp.distributions.Distribution; distribution object with a log_prob method.
    :return: numpy.array; log-likelihood of the ground truth.
    """
    return -y_hat.log_prob(y_true)

########################################################################################################################


def train_prob_regressor(data_path: str,
                         num_prods: int,
                         test_split: float,
                         seed: int,
                         model_savepath: str) -> PoissonNeuralNetwork:
    """
    Train a probabilistic regressor maximizing the log-likelihood.
    :param data_path: str; where the instances are loaded from.
    :param num_prods: int; the number of products in the MSC instances.
    :param test_split: float; fraction of the instances used for test.
    :param seed: int; seed to ensure reproducibility results.
    :param model_savepath: str; where the model is saved to.
    :return PoissonNeuralNetwork; the fitted probabilistic regressor.
    """

    # Load the instances
    instances = list()

    # Load the Set Multi-cover instances
    print('Loading instances...')
    for f in os.listdir(data_path):
        if os.path.isdir(os.path.join(data_path, f)):
            instances.append(load_msc(os.path.join(data_path, f, 'instance.pkl')))
    print('Finished')

    # Split between training and test instances
    train_instances, test_instances = \
        train_test_split(instances, test_size=test_split, random_state=seed)
    train_obs = np.asarray([inst.observables for inst in train_instances])
    test_obs = np.asarray([inst.observables for inst in test_instances])
    train_obs = np.expand_dims(train_obs, axis=1)
    test_obs = np.expand_dims(test_obs, axis=1)
    train_lambdas = np.asarray([inst.lmbds for inst in train_instances])
    test_lambdas = np.asarray([inst.lmbds for inst in test_instances])

    # Create and train the probabilistic regressor model
    model = PoissonNeuralNetwork(input_shape=(1,), output_shape=num_prods)
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

    # Save model and results
    model.save(model_savepath)

    # Evaluate the probablistic regressor
    preds = model(test_obs)
    lmbds_hat = preds.rate.numpy()
    maes = list()
    for i in range(num_prods):
        '''plt.scatter(test_lambdas[:, i], lmbds_hat[:, i], label='Predicted lambda')
        plt.plot(test_lambdas[:, i], test_lambdas[:, i], label='True lambda', color='r')
        plt.xlabel('True lambda')
        plt.legend()
        plt.title(f'Product {i + 1}')
        plt.savefig(os.path.join(MODEL_SAVEPATH, f'Product {i + 1}.png'))
        plt.close('all')'''
        mae = np.mean(np.abs(lmbds_hat[:, i] - test_lambdas[:, i]))
        maes.append(mae)

    print(f'Max MAE: {np.max(maes)} | Min MAE: {np.min(maes)} | Mean MAE: {np.mean(maes)}')

    return model

########################################################################################################################


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("datadir", type=str, help="Data directory")
    parser.add_argument("modeldir", type=str, help="Model save directory")
    parser.add_argument("resdir", type=str, help="Results directory")
    parser.add_argument("--seed", type=int, help="Seed to ensure reproducibility of the results")
    parser.add_argument("--num-prods", type=int, help="Number of products")
    parser.add_argument("--num-sets", type=int, help="Number of sets")
    parser.add_argument("--num-instances", type=int, help="Number of generated instances")
    parser.add_argument("--evaluated-instances", type=int, help="Number of test set instances used for the evaluation")
    parser.add_argument("--test-split", type=float, help="Fraction of the instances used as test")
    parser.add_argument("--num-scenarios", type=int, nargs='+', help="Number of scenarios for the stochastic algo")
    parser.add_argument("--train", action='store_true', help="Set this flag if you want to train the probabilistic")
    args = parser.parse_args()

    SEED = int(args.seed)
    NUM_PRODS = int(args.num_prods)
    NUM_SETS = int(args.num_sets)
    NUM_INSTANCES = int(args.num_instances)
    NUM_EVALUATED_INSTANCES = int(args.evaluated_instances)
    TEST_SPLIT = float(args.test_split)
    TRAIN = args.train
    NUM_SCENARIOS = args.num_scenarios
    DATA_PATH = args.datadir
    DATA_PATH = os.path.join(DATA_PATH,
                             f'{NUM_PRODS}x{NUM_SETS}',
                             'linear',
                             f'{NUM_INSTANCES}-instances',
                             f'seed-{SEED}')
    MODEL_SAVEPATH = args.modeldir
    MODEL_SAVEPATH = os.path.join(MODEL_SAVEPATH,
                                  f'{NUM_PRODS}x{NUM_SETS}',
                                  'linear',
                                  f'{NUM_INSTANCES}-instances',
                                  f'seed-{SEED}')
    LOG_DIR = args.resdir
    LOG_DIR = os.path.join(LOG_DIR,
                           f'{NUM_PRODS}x{NUM_SETS}',
                           'linear',
                           f'{NUM_INSTANCES}-instances',
                           f'seed-{SEED}')

    # Create the model savepath if it does not exist
    if not os.path.exists(MODEL_SAVEPATH):
        os.makedirs(MODEL_SAVEPATH)

    # Set the random seed to ensure reproducibility
    np.random.seed(SEED)

    # Create the probabilistic regressor
    model = PoissonNeuralNetwork(input_shape=(1, ), output_shape=NUM_PRODS)

    # Train the probabilistic regressor from scratch or load it from file
    if TRAIN:
        model = train_prob_regressor(data_path=DATA_PATH,
                                     num_prods=NUM_PRODS,
                                     test_split=TEST_SPLIT,
                                     seed=SEED,
                                     model_savepath=MODEL_SAVEPATH)
    else:
        model.load(MODEL_SAVEPATH)

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
        _, test_optimal_costs = \
            train_test_split(instances, optimal_costs, test_size=TEST_SPLIT, random_state=SEED)

    # Evaluate the probabilistic model
    test_obs = np.asarray([inst.observables for inst in test_instances])
    test_obs = np.expand_dims(test_obs, axis=1)
    test_lambdas = np.asarray([inst.lmbds for inst in test_instances])

    preds = model(test_obs)
    lmbds_hat = preds.rate.numpy()
    mapes = list()
    for i in range(NUM_PRODS):
        mape = mean_absolute_percentage_error(lmbds_hat[:, i], test_lambdas[:, i])
        mapes.append(mape)

    mapes = np.asarray(mapes)
    np.save(os.path.join(LOG_DIR, 'mapes'), mapes)

    print(f'Max MAE: {np.max(mapes)} | Min MAE: {np.min(mapes)} | Mean MAE: {np.mean(mapes)}')

    # Keep track of runtimes and cost grouped by number of scenarios
    runtimes = dict()
    costs = dict()

    # Iterate over all the test instances and evaluate the predict-then-optimize approach
    for inst_idx, inst in enumerate(test_instances[:NUM_EVALUATED_INSTANCES]):

        # Observable for the current instance
        obs = inst.observables
        # FIXME: we assume the observation to be univariate
        obs = np.array([[obs]])

        # These are the instances sampled from the ML model and given as input to the stochastic method
        train_instances = list()
        for idx in range(max(NUM_SCENARIOS)):

            msc_copy = copy.deepcopy(inst)
            poisson = model(obs)
            # The impose the first scenario to have demands equal to the Poisson rates
            if idx == 0:
                rates = poisson.rate.numpy()
                demands = rates.astype(dtype=np.int32)
            else:
                demands = poisson.sample().numpy()
            demands = np.squeeze(demands)
            msc_copy.demands = demands
            train_instances.append(msc_copy)

        # Apply the stochastic algo with the user defined number of scenarios
        for num_scenarios in NUM_SCENARIOS:
            if num_scenarios not in runtimes.keys():
                runtimes[num_scenarios] = list()
            if num_scenarios not in costs.keys():
                costs[num_scenarios] = list()

            # Create the stochastic algorithm instance and solve the problem
            instance = StochasticAlgorithm(instances=train_instances, num_scenarios=num_scenarios)
            res = instance.solve()
            runtimes[num_scenarios].append(res['Runtime'])

            # Evaluate the solution found on the current instance
            print(f'[Predict-then-optimize] Results for instance n. {inst_idx+1}/{NUM_EVALUATED_INSTANCES}')
            res = evaluate_stochastic_algo(res=res,
                                           num_scenarios=num_scenarios,
                                           instances=[inst],
                                           optimal_costs=test_optimal_costs)

            costs[num_scenarios].append(res['Stochastic algo mean cost'])

    # Save results on file
    # Remove the logging folder if exists or create it
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    else:
        shutil.rmtree(LOG_DIR)
        os.makedirs(LOG_DIR)

    pickle.dump(runtimes, open(os.path.join(LOG_DIR, 'runtimes.pkl'), 'wb'))
    pickle.dump(costs, open(os.path.join(LOG_DIR, 'costs.pkl'), 'wb'))
