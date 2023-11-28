"""
    Utility methods to implement a custom evaluation in the garage library.
"""

import numpy as np
import copy
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
from dowel import tabular, logger
import time
from garage.np import discount_cumsum
from garage import StepType
from garage.trainer import Trainer, NotSetupError, TrainArgs
from garage import EnvSpec
from garage.envs.gym_env import GymEnv

########################################################################################################################


def log_performance(itr, batch, discount, prefix='Evaluation'):
    """Evaluate the performance of an algorithm on a batch of episodes.
       Custom implementation of the log_performance method of garage
       (https://github.com/rlworkgroup/garage/blob/master/src/garage/_functions.py).
    Args:
        itr (int): Iteration number.
        batch (EpisodeBatch): The episodes to evaluate with.
        discount (float): Discount value, from algorithm's property.
        prefix (str): Prefix to add to all logged keys.
    Returns:
        numpy.ndarray: Undiscounted returns.
    """

    returns = []
    undiscounted_returns = []
    termination = []
    success = []
    batch_feasibles = list()
    batch_true_cost = list()

    for eps in batch.split():
        returns.append(discount_cumsum(eps.rewards, discount))
        undiscounted_returns.append(sum(eps.rewards))

        assert 'feasible' in eps.env_infos.keys(), "A feasible value is expected"
        assert 'true cost' in eps.env_infos.keys(), "A true cost value is expected"

        successful = np.sum(~eps.env_infos['feasible']) == 0
        batch_feasibles.append(successful)
        batch_true_cost.append(eps.env_infos['true cost'][-1])

        termination.append(
            float(
                any(step_type == StepType.TERMINAL
                    for step_type in eps.step_types)))
        if 'success' in eps.env_infos:
            success.append(float(eps.env_infos['success'].any()))

    average_discounted_return = np.mean([rtn[0] for rtn in returns])
    average_feasibility_ratio = np.mean(batch_feasibles)
    average_true_cost = np.mean(batch_true_cost)

    with tabular.prefix(prefix + '/'):
        tabular.record('Iteration', itr)
        tabular.record('NumEpisodes', len(returns))
        tabular.record('AverageDiscountedReturn', average_discounted_return)
        tabular.record('AverageReturn', np.mean(undiscounted_returns))

        # NOTE: add additional evaluation metrics
        tabular.record('BatchAvgFeasibilityRatio', np.mean(average_feasibility_ratio))
        tabular.record('BatchAvgTrueCost', np.mean(average_true_cost))

        tabular.record('StdReturn', np.std(undiscounted_returns))
        tabular.record('MaxReturn', np.max(undiscounted_returns))
        tabular.record('MinReturn', np.min(undiscounted_returns))
        tabular.record('TerminationRate', np.mean(termination))
        if success:
            tabular.record('SuccessRate', np.mean(success))

    return undiscounted_returns

########################################################################################################################


class CustomTrainer(Trainer):
    """
    Custom implementation of the garage.trainer.Trainer to prevent avoid serialization.
    (https://github.com/rlworkgroup/garage/blob/3492f446633a7e748f2f79077f6301c5b3ec9281/src/garage/trainer.py)
    """
    def __init__(self, snapshot_config):
        super().__init__(snapshot_config=snapshot_config)

    def train(self,
              n_epochs,
              batch_size=None,
              plot=False,
              store_episodes=False,
              pause_for_plot=False):
        """Start training.
        Args:
            n_epochs (int): Number of epochs.
            batch_size (int or None): Number of environment steps in one batch.
            plot (bool): Visualize an episode from the policy after each epoch.
            store_episodes (bool): Save episodes in snapshot.
            pause_for_plot (bool): Pause for plot.
        Raises:
            NotSetupError: If train() is called before setup().
        Returns:
            float: The average return in last epoch cycle.
        """
        if not self._has_setup:
            raise NotSetupError(
                'Use setup() to setup trainer before training.')

        # Save arguments for restore
        self._train_args = TrainArgs(n_epochs=n_epochs,
                                     batch_size=batch_size,
                                     plot=plot,
                                     store_episodes=store_episodes,
                                     pause_for_plot=pause_for_plot,
                                     start_epoch=0)

        self._plot = plot
        self._start_worker()

        average_return = self._algo.train(self)
        self._shutdown_worker()

        return average_return

    def save(self, epoch):
        """Save snapshot of current batch.
        Args:
            epoch (int): Epoch.
        Raises:
            NotSetupError: if save() is called before the trainer is set up.
        """
        if not self._has_setup:
            raise NotSetupError('Use setup() to setup trainer before saving.')

        logger.log('Saving snapshot...')

        params = dict()
        # Save arguments
        # params['seed'] = self._seed
        # params['train_args'] = self._train_args
        # params['stats'] = self._stats

        # Save states
        params['algo'] = self._algo.policy
        # params['n_workers'] = self._n_workers
        # params['worker_class'] = self._worker_class
        # params['worker_args'] = self._worker_args

        start = time.time()
        self._snapshotter.save_itr_params(epoch, params)
        end = time.time()
        print(end - start)

        logger.log('Saved')

########################################################################################################################


class CustomTFTrainer(CustomTrainer):
    """
    Custom implementation of garage.trainer.TFTrainer to prevent environment serialization
    (https://github.com/rlworkgroup/garage/blob/3492f446633a7e748f2f79077f6301c5b3ec9281/src/garage/trainer.py)
    This class implements a trainer for TensorFlow algorithms.
    A trainer provides a default TensorFlow session using python context.
    This is useful for those experiment components (e.g. policy) that require a
    TensorFlow session during construction.
    Use trainer.setup(algo, env) to setup algorithm and environment for trainer
    and trainer.train() to start training.
    Args:
        snapshot_config (garage.experiment.SnapshotConfig): The snapshot
            configuration used by Trainer to create the snapshotter.
            If None, it will create one with default settings.
        sess (tf.Session): An optional TensorFlow session.
              A new session will be created immediately if not provided.
    Note:
        When resume via command line, new snapshots will be
        saved into the SAME directory if not specified.
        When resume programmatically, snapshot directory should be
        specify manually or through @wrap_experiment interface.
    Examples:
        # to train
        with TFTrainer() as trainer:
            env = gym.make('CartPole-v1')
            policy = CategoricalMLPPolicy(
                env_spec=env.spec,
                hidden_sizes=(32, 32))
            algo = TRPO(
                env=env,
                policy=policy,
                baseline=baseline,
                max_episode_length=100,
                discount=0.99,
                max_kl_step=0.01)
            trainer.setup(algo, env)
            trainer.train(n_epochs=100, batch_size=4000)
        # to resume immediately.
        with TFTrainer() as trainer:
            trainer.restore(resume_from_dir)
            trainer.resume()
        # to resume with modified training arguments.
        with TFTrainer() as trainer:
            trainer.restore(resume_from_dir)
            trainer.resume(n_epochs=20)
    """

    def __init__(self, snapshot_config, sess=None):
        # pylint: disable=import-outside-toplevel
        import tensorflow
        # pylint: disable=global-statement
        global tf
        tf = tensorflow
        super().__init__(snapshot_config=snapshot_config)
        self.sess = sess or tf.compat.v1.Session()
        self.sess_entered = False

    def __enter__(self):
        """Set self.sess as the default session.
        Returns:
            TFTrainer: This trainer.
        """
        if tf.compat.v1.get_default_session() is not self.sess:
            self.sess.__enter__()
            self.sess_entered = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Leave session.
        Args:
            exc_type (str): Type.
            exc_val (object): Value.
            exc_tb (object): Traceback.
        """
        if tf.compat.v1.get_default_session(
        ) is self.sess and self.sess_entered:
            self.sess.__exit__(exc_type, exc_val, exc_tb)
            self.sess_entered = False

    # NOTE: make the environment accessible from the outside
    @property
    def env(self):
        return self._env

    def setup(self, algo, env):
        """Set up trainer and sessions for algorithm and environment.
        This method saves algo and env within trainer and creates a sampler,
        and initializes all uninitialized variables in session.
        Note:
            After setup() is called all variables in session should have been
            initialized. setup() respects existing values in session so
            policy weights can be loaded before setup().
        Args:
            algo (RLAlgorithm): An algorithm instance.
            env (Environment): An environment instance.
        """
        self.initialize_tf_vars()
        logger.log(self.sess.graph)
        super().setup(algo, env)

    def _start_worker(self):
        """Start Plotter and Sampler workers."""
        self._sampler.start_worker()
        if self._plot:
            # pylint: disable=import-outside-toplevel
            from garage.tf.plotter import Plotter
            self._plotter = Plotter(self.get_env_copy(),
                                    self._algo.policy,
                                    sess=tf.compat.v1.get_default_session())
            self._plotter.start()

    def initialize_tf_vars(self):
        """Initialize all uninitialized variables in session."""
        with tf.name_scope('initialize_tf_vars'):
            uninited_set = [
                e.decode() for e in self.sess.run(
                    tf.compat.v1.report_uninitialized_variables())
            ]
            self.sess.run(
                tf.compat.v1.variables_initializer([
                    v for v in tf.compat.v1.global_variables()
                    if v.name.split(':')[0] in uninited_set
                ]))

########################################################################################################################


@dataclass(frozen=True, init=False)
class CustomEnvSpec(EnvSpec):
    """
       This class extends the garage.EnvSpec class adding a demands scaler attribute.
       Describes the observations, actions, and time horizon of an MDP.
       Args:
           observation_space (akro.Space): The observation space of the env.
           action_space (akro.Space): The action space of the env.
           max_episode_length (int): The maximum number of steps allowed in an
               episode.
       """
    def __init__(self,
                 observation_space,
                 action_space,
                 scaler,
                 max_episode_length=None):

        object.__setattr__(self, 'scaler', scaler)

        super().__init__(action_space=action_space,
                         observation_space=observation_space,
                         max_episode_length=max_episode_length)

        scaler: StandardScaler

########################################################################################################################


class CustomEnv(GymEnv):
    """
    This class extends the garage.envs.GymEnv class adding the demands scaler attribute.
    """
    def __init__(self, env, is_image=False, max_episode_length=None):

        assert hasattr(env, 'demands_scaler'), "The environvment must have a demands_scaler attribute"

        super().__init__(env=env, is_image=is_image, max_episode_length=max_episode_length)

        self._spec = CustomEnvSpec(action_space=self.action_space,
                                   observation_space=self.observation_space,
                                   max_episode_length=self._max_episode_length,
                                   scaler=env.demands_scaler)

    @property
    def env(self):
        return self._env