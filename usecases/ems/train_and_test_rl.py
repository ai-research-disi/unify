"""
    Main methods to train and test the different approaches.
"""

import numpy as np
import pandas as pd
import garage
from garage.tf.baselines import ContinuousMLPBaseline
from garage.sampler import LocalSampler
from garage.tf.policies import GaussianMLPPolicy
from garage.trainer import TFTrainer
from garage import wrap_experiment
from garage.experiment import Snapshotter
import tensorflow as tf
import os
import argparse
from typing import Union, List
from helpers.utility import timestamps_headers, my_wrap_experiment, renamed_load
from helpers.utility import METHODS, MODES
from usecases.ems.rl.utility import CustomTFTrainer
from usecases.ems.rl.algos import VPG
from usecases.ems.vpp_envs import make_env

########################################################################################################################


def train_rl_algo(ctxt: garage.experiment.SnapshotConfig = None,
                  method: str = None,
                  test_split: Union[float, List[int]] = 0.25,
                  num_epochs: int = 1000,
                  noise_std_dev: Union[float, int] = 0.01,
                  batch_size: int = 100,
                  safety_layer: bool = False,
                  predictions_filepath: str = None,
                  prices_filepath: str = None,
                  shifts_filepath: str = None):
    """
    Training routing.
    :param ctxt: garage.experiment.SnapshotConfig; the snapshot configuration used by Trainer to create the snapshotter.
                                                   If None, one will be created with default settings.
    :param method: string; choose among one of the available methods.
    :param test_split: float or list of int; fraction or indexes of the instances to be used for test.
    :param num_epochs: int; number of training epochs.
    :param noise_std_dev: float; standard deviation for the additive gaussian noise.
    :param batch_size: int; batch size.
    :param safety_layer: bool; True if you want to use the safety layer; False otherwise. The safety layer is available
                               for the full RL MDP version of the environment.
    :param predictions_filepath: str; where the forecast instances are loaded from.
    :param prices_filepath: str; where the prices of the electricity market are loaded from.
    :param shifts_filepath: str; where the optimal dayahead shifts are loaded from.
    :return:
    """

    # Default datapath if not provided by the user
    if predictions_filepath is None:
        predictions_filepath = os.path.join('data', 'ems', 'Dataset10k.csv')
    if prices_filepath is None:
        prices_filepath = os.path.join('data', 'ems', 'gmePrices.npy')
    if shifts_filepath is None:
        shifts_filepath = os.path.join('data', 'ems', 'optShift.npy')

    # Check that the selected method is valid
    assert method in METHODS, f"{method} is not valid"
    print(f'Selected method: {method}')

    # A trainer provides a default TensorFlow session using python context
    with CustomTFTrainer(snapshot_config=ctxt) as trainer:

        # Load data from file
        # Check that all the required files exist
        assert os.path.isfile(predictions_filepath), f"{predictions_filepath} does not exist"
        assert os.path.isfile(prices_filepath), f"{prices_filepath} does not exist"
        assert os.path.isfile(shifts_filepath), f"{shifts_filepath} does not exist"
        predictions = pd.read_csv(predictions_filepath)
        shift = np.load(shifts_filepath)
        c_grid = np.load(prices_filepath)

        # Split between training and test
        if isinstance(test_split, float):
            split_index = int(len(predictions) * (1 - test_split))
            train_predictions = predictions[:split_index]
        elif isinstance(test_split, list):
            split_index = test_split
            train_predictions = predictions.iloc[split_index]
        else:
            raise Exception("test_split must be list of int or float")

        # Create the environment
        env, discount, max_episode_length = \
            make_env(method=method,
                     predictions=train_predictions,
                     shift=shift,
                     c_grid=c_grid,
                     noise_std_dev=noise_std_dev,
                     safety_layer=safety_layer)

        # A policy represented by a Gaussian distribution which is parameterized by a multilayer perceptron (MLP)
        policy = GaussianMLPPolicy(env.spec)
        obs, _ = env.reset()

        # A value function using a MLP network.
        baseline = ContinuousMLPBaseline(env_spec=env.spec)

        # It's called the "Local" sampler because it runs everything in the same process and thread as where
        # it was called from.
        sampler = LocalSampler(agents=policy,
                               envs=env,
                               max_episode_length=max_episode_length,
                               is_tf_worker=True)

        # Vanilla Policy Gradient
        algo = VPG(env_spec=env.spec,
                   baseline=baseline,
                   policy=policy,
                   sampler=sampler,
                   discount=discount,
                   optimizer_args=dict(learning_rate=0.01, ))

        trainer.setup(algo, env)
        trainer.train(n_epochs=num_epochs, batch_size=batch_size, plot=False)

########################################################################################################################


def test_rl_algo(log_dir: str,
                 predictions_filepath: str,
                 shifts_filepath: str,
                 prices_filepath: str,
                 method: str,
                 test_split: Union[float, List[int]],
                 num_episodes: int = 100,
                 safety_layer: bool = False):
    """
    Test a trained agent.
    :param log_dir: string; path where training information are saved to.
    :param predictions_filepath: string; where instances are loaded from.
    :param shifts_filepath: string; where optimal shifts are loaded from.
    :param prices_filepath: string; where prices are loaded from.
    :param method: string; choose among one of the available methods.
    :param test_split: float or list of int; fraction of the instances or list of instances indexes.
    :param num_episodes: int; number of episodes.
    :param safety_layer: bool; True if you want to use the safety layer; False otherwise. The safety layer is available
                               for the full RL MDP version of the environment.
    :return:
    """

    # Create TF1 session and load all the experiments data
    tf.compat.v1.disable_eager_execution()
    tf.compat.v1.reset_default_graph()
    with tf.compat.v1.Session() as _:
        # Load parameters and get the agent
        data = renamed_load(open(os.path.join(log_dir, 'params.pkl'), 'rb'))
        policy = data['algo']

        # Load data from file
        # Check that all the required files exist
        assert os.path.isfile(predictions_filepath), f"{predictions_filepath} does not exist"
        assert os.path.isfile(prices_filepath), f"{prices_filepath} does not exist"
        assert os.path.isfile(shifts_filepath), f"{shifts_filepath} does not exist"
        predictions = pd.read_csv(predictions_filepath)
        shift = np.load(shifts_filepath)
        c_grid = np.load(prices_filepath)

        # Split between training and test
        if isinstance(test_split, float):
            split_index = int(len(predictions) * (1 - test_split))
            train_predictions = predictions[:split_index]
        elif isinstance(test_split, list):
            split_index = test_split
            train_predictions = predictions.iloc[split_index]
        else:
            raise Exception("test_split must be list of int or float")

        # Create the environment
        env, discount, max_episode_length = \
            make_env(method=method,
                     predictions=train_predictions,
                     shift=shift,
                     c_grid=c_grid,
                     noise_std_dev=0,
                     safety_layer=safety_layer)

        timestamps = timestamps_headers(env.n)
        all_rewards = list()

        total_reward = 0

        # Loop for each episode
        for episode in range(num_episodes):
            last_obs, _ = env.reset()
            done = False

            episode_reward = 0

            all_actions = []

            # Perform an episode
            while not done:
                # env.render(mode='ascii')
                if hasattr(policy, 'get_action'):
                    policy = policy
                elif hasattr(policy, 'policy'):
                    policy = policy.policy
                else:
                    raise Exception("The policy object must have either 'policy' or 'get_action' attribute")

                _, agent_info = policy.get_action(last_obs)
                a = agent_info['mean']
                all_actions.append(np.squeeze(a))

                step = env.step(a)

                total_reward -= step.reward
                episode_reward -= step.reward

                if step.terminal or step.timeout:
                    break
                last_obs = step.observation

            print(f'\nTotal reward: {episode_reward}')
            all_rewards.append(episode_reward)

            if method == 'rl-mdp':
                all_actions = np.expand_dims(all_actions, axis=0)

        if 'rl' in method:
            action_save_name = 'solution'
        else:
            action_save_name = 'cvirt'

        # Save the agent's actions
        all_actions = np.squeeze(all_actions)
        np.save(os.path.join(log_dir, action_save_name), all_actions)

########################################################################################################################


@wrap_experiment
def resume_experiment(ctxt, saved_dir):
    """Resume a Tensorflow experiment.
    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by Trainer to create the snapshotter.
        saved_dir (str): Path where snapshots are saved.
    """
    with TFTrainer(snapshot_config=ctxt) as trainer:
        trainer.restore(from_dir=saved_dir)
        trainer.resume()

########################################################################################################################


def main():
    """
    Main execution function for training and test the algorithms on the EMS use cases.
    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("logdir", type=str, help="Logging directory")
    parser.add_argument("--method",
                        type=str,
                        choices=METHODS,
                        help="'unify-all-at-once': as described in the paper;"
                             + "'unify-sequential': as described in the paper;"
                             + "'rl-all-at-once': end-to-end RL approach which directly provides the decision "
                             + "variables for all the stages;"
                             + "'rl-sequential': this is referred to as 'rl' in the paper.")
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, help="Batch size")
    parser.add_argument("--mode",
                        type=str,
                        choices=MODES,
                        required=True,
                        help="'train': if you want to train a model from scratch;"
                             + "'test': if you want to test an existing model.")
    parser.add_argument("--safety-layer", action='store_true')
    args = parser.parse_args()

    # Extract some arguments
    LOG_DIR = args.logdir
    METHOD = args.method
    SAFETY_LAYER = args.safety_layer
    mode = args.mode

    if mode == 'train':
        EPOCHS = args.epochs
        BATCH_SIZE = args.batch_size

    # Randomly choose 100 instances
    np.random.seed(0)
    indexes = np.arange(10000, dtype=np.int32)
    indexes = np.random.choice(indexes, size=100)

    print(indexes)

    if mode == 'train':
        # Training routine
        for instance_idx in indexes:
            tf.compat.v1.disable_eager_execution()
            tf.compat.v1.reset_default_graph()
            run = my_wrap_experiment(train_rl_algo,
                                     logging_dir=f'{LOG_DIR}_{instance_idx}',
                                     snapshot_mode='last',
                                     snapshot_gap=1,
                                     archive_launch_repo=False)

            run(method=METHOD,
                test_split=[instance_idx],
                num_epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                noise_std_dev=0.01,
                safety_layer=SAFETY_LAYER)

    elif mode == 'test':
        # Test trained methods
        for instance_idx in indexes:
            test_rl_algo(log_dir=f'{LOG_DIR}_{instance_idx}',
                         predictions_filepath=os.path.join('data', 'ems', 'Dataset10k.csv'),
                         shifts_filepath=os.path.join('data', 'ems', 'optShift.npy'),
                         prices_filepath=os.path.join('data', 'ems', 'gmePrices.npy'),
                         method=METHOD,
                         test_split=[instance_idx],
                         num_episodes=1,
                         safety_layer=SAFETY_LAYER)

    else:
        raise Exception(f"{mode} is not supported".format(mode))
