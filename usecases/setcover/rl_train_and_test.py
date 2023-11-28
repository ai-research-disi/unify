"""
    Training and testing methods for the hybrid RL+CO formulation.
"""

from garage.tf.baselines import ContinuousMLPBaseline
from garage.sampler import LocalSampler
from usecases.setcover.rl.algos import VPG
from garage.tf.policies import GaussianMLPPolicy
from garage.tf.policies import Policy
from garage.experiment import SnapshotConfig
import tensorflow as tf
import os
import pickle5
import pickle
import numpy as np
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import time
from usecases.setcover.generate_instances import MinSetCoverEnv
from usecases.setcover.rl.utility import CustomTFTrainer, CustomEnv
from helpers.utility import renamed_load, my_wrap_experiment


########################################################################################################################

def _evaluate_model(env: MinSetCoverEnv,
                    policy: Policy,
                    log_dir: str,
                    mode: str,
                    num_episodes: int = None,
                    plot: bool = False):
    """
    Evaluate the policy in the environment and save results in the logging directory.
    :param env: MinSetCoverEnv; the environment.
    :param policy: garage.tf.policy; the RL policy.
    :param log_dir: string; the logging directory path.
    :param mode: string; choose between evaluation on training or test sets.
    :param num_episodes: int; number of evaluation episodes.
    :param plot: bool; if True, plot the predicted demands.
    :return:
    """

    assert mode in ['train', 'test'], "Supported modes are 'train' or 'test'"

    if mode == 'train':
        instances = env.train_instances
        prefix = 'Training instances'
        save_prefix = 'train'
    else:
        instances = env.test_instances
        prefix = 'Test instances'
        save_prefix = 'test'

    total_reward = 0

    # Keep track of required values
    all_actions = list()
    all_obs = list()
    all_lambdas = list()
    all_rewards = list()
    all_episode_times = list()

    # Loop for each episode

    if num_episodes is None:
        num_episodes = len(instances)

    for episode in range(num_episodes):

        print(f'{prefix} - Episode {episode + 1}/{num_episodes}')

        env._current_instance = instances[episode]
        all_obs.append(env.current_instance.observables)
        all_lambdas.append(env.current_instance.lmbds)
        done = False

        episode_reward = 0
        episode_time = 0

        # Perform an episode
        while not done:
            # env.render(mode='ascii')
            # Keep track of time

            start = time.time()
            _, agent_info = policy.get_action(env.current_instance.observables)
            end = time.time()
            elapsed = end - start
            episode_time += elapsed

            mean_action = agent_info['mean']
            std_action = np.exp(agent_info['log_std'])
            a = mean_action

            start = time.time()
            state, reward, done, info = env.step(a)
            end = time.time()
            elapsed = end - start
            episode_time += elapsed

            all_actions.append(info['Action'])

            total_reward -= reward
            episode_reward -= reward

            if done:
                all_episode_times.append(episode_time)
                break

        all_rewards.append(episode_reward)
        print(f'Episode reward: {episode_reward}')
        print(f'Runtime: {episode_time}')

    # Visualize and save results
    print(f'Mean cost: {np.mean(all_rewards)} | Std dev cost: {np.std(all_rewards)} | ' + \
          f'Mean episode time: {np.mean(all_episode_times)}')

    res = dict()
    res['Mean cost'] = np.mean(all_rewards)
    res['Std dev'] = np.std(all_rewards)
    res['Mean runtime'] = np.mean(all_episode_times)
    res['Std runtime'] = np.std(all_episode_times)
    pickle.dump(res, open(os.path.join(log_dir, f'{save_prefix}-set-res.pkl'), 'wb'))

    all_actions = np.asarray(all_actions)
    all_obs = np.asarray(all_obs)
    all_lambdas = np.asarray(all_lambdas)

    if plot:
        sns.set_style('darkgrid')
        # Plot the relationship between the observations and the actions/lambda values
        for prod in range(env.current_instance.num_products):
            plt.scatter(all_obs, all_actions[:, prod], label='Predicted demands')
            plt.scatter(all_obs, all_lambdas[:, prod], label='Lambda')
            plt.xlabel('Observable')
            plt.legend()
            plt.savefig(f'Product #{prod + 1}.png')
            plt.close('all')

########################################################################################################################


def train(ctxt: SnapshotConfig = None,
          num_epochs: int = 100,
          batch_size: int = 100,
          env: MinSetCoverEnv = None):
    """
    :param ctxt: garage.experiment.SnapshotConfig: The snapshot configuration used by Trainer to create the
                                                   snapshotter.
    :param num_epochs: int; number of training epochs.
    :param batch_size: int; batch size.
    :param env: usecases.setcover.generate_instances.MinSetCoverEnv; the Minimum Set Cover environment instance.
    :return:
    """

    # Check that the env is not None
    assert env is not None

    # A trainer provides a default TensorFlow session using python context
    with CustomTFTrainer(snapshot_config=ctxt) as trainer:

        # Garage wrapping of a gym environment
        env = CustomEnv(env, max_episode_length=1)

        # A policy represented by a Gaussian distribution which is parameterized by a multilayer perceptron (MLP)
        policy = GaussianMLPPolicy(env.spec)
        obs, _ = env.reset()

        # A value function using a MLP network.
        baseline = ContinuousMLPBaseline(env_spec=env.spec)

        # It's called the "Local" sampler because it runs everything in the same process and thread as where
        # it was called from.
        sampler = LocalSampler(agents=policy,
                               envs=env,
                               max_episode_length=1,
                               is_tf_worker=True)

        # Vanilla Policy Gradient
        algo = VPG(env_spec=env.spec,
                   baseline=baseline,
                   policy=policy,
                   sampler=sampler,
                   discount=0.99,
                   optimizer_args=dict(learning_rate=0.001, ))

        trainer.setup(algo, env)
        trainer.train(n_epochs=num_epochs, batch_size=batch_size, plot=False)


########################################################################################################################


def test_rl_algo(log_dir: str,
                 env: MinSetCoverEnv,
                 num_episodes: int):
    """
    Test an already trained agent.
    :param log_dir: string; loadpath of the agent.
    :param env; MinSetCoverEnv; the environment on which the agent is evaluated.
    :param num_episodes: int; number of evaluation episodes.
    :return:
    """
    # Create TF1 session and load all the experiments data
    tf.compat.v1.disable_eager_execution()
    tf.compat.v1.reset_default_graph()
    with tf.compat.v1.Session() as _:
        # Load parameters
        data = renamed_load(open(os.path.join(log_dir, 'params.pkl'), 'rb'))
        # Get the agent
        algo = data['algo']

        # Get the policy
        policy = algo

        _evaluate_model(env=env, policy=policy, log_dir=log_dir, mode='train')
        _evaluate_model(env=env, policy=policy, log_dir=log_dir, mode='test', num_episodes=num_episodes)

########################################################################################################################


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("datadir", type=str, help="Data directory")
    parser.add_argument("resdir", type=str, help="Results directory")
    parser.add_argument("--num-prods", type=int, help="Number of products")
    parser.add_argument("--num-sets", type=int, help="Number of sets")
    parser.add_argument("--batch-size", type=int, help="Batch size")
    parser.add_argument("--num-epochs", type=int, help="Number of epochs")
    parser.add_argument("--seed", type=int, help="Seed to ensure reproducibility of the results")
    parser.add_argument("--num-instances", type=int, help="Number of MSC instances")
    parser.add_argument("--evaluated-instances", type=int, help="Number of test set instances used for the evaluation")
    parser.add_argument("--train", action='store_true', help="Set this flag if you want to train the probabilistic")

    args = parser.parse_args()

    # Set some constant values
    NUM_PRODS = int(args.num_prods)
    NUM_SETS = int(args.num_sets)
    BATCH_SIZE = int(args.batch_size)
    EPOCHS = int(args.num_epochs)
    SEED = int(args.seed)
    NUM_INSTANCES = int(args.num_instances)
    NUM_EVALUATED_INSTANCES = int(args.evaluated_instances)
    TRAIN = args.train
    SAVEPATH_PREFIX = args.resdir
    SAVEPATH_PREFIX = os.path.join(SAVEPATH_PREFIX,
                                   f'{NUM_PRODS}x{NUM_SETS}',
                                   'linear',
                                   f'{NUM_INSTANCES}-instances',
                                   f'seed-{SEED}')
    DATA_PATH = args.datadir
    DATA_PATH = os.path.join(DATA_PATH,
                             f'{NUM_PRODS}x{NUM_SETS}',
                             'linear',
                             f'{NUM_INSTANCES}-instances',
                             f'seed-{SEED}')

    # Set the random seed to ensure reproducibility
    np.random.seed(SEED)

    # Create the environment
    env = MinSetCoverEnv(num_prods=NUM_PRODS,
                         num_sets=NUM_SETS,
                         instances_filepath=DATA_PATH,
                         seed=SEED)

    # Create the saving directory if it does not exist
    if not os.path.exists(SAVEPATH_PREFIX):
        os.makedirs(SAVEPATH_PREFIX)

    # Train and test the RL algo
    tf.compat.v1.disable_eager_execution()
    tf.compat.v1.reset_default_graph()
    if TRAIN:
        run = my_wrap_experiment(train,
                                 logging_dir=SAVEPATH_PREFIX,
                                 snapshot_mode='gap_overwrite',
                                 snapshot_gap=EPOCHS // 10,
                                 # FIXME: archive_launch_repo=True is not supported
                                 archive_launch_repo=False)
        run(num_epochs=EPOCHS, batch_size=BATCH_SIZE, env=env)

    test_rl_algo(SAVEPATH_PREFIX, env, num_episodes=NUM_EVALUATED_INSTANCES)
