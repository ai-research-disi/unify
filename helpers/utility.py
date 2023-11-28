"""
    Utility methods.
"""

from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import io
import sys
import functools
import gc
import collections
import os
import subprocess
import dowel
from dowel import logger
from typing import Tuple, Union, List
from garage.experiment.experiment import ExperimentTemplate, ExperimentContext
from garage.experiment.experiment import dump_json, make_launcher_archive
from garage.experiment.experiment import _make_sequential_log_dir, _make_experiment_signature

import __main__ as main

########################################################################################################################

# Definition of some global variables of the script

# The serialization of garage is quite deep and keep tracks of the imported modules; if their path or names change, you
# need to insert them in this dictionary of legacy modules
LEGACY_MODULES = {"rl_utils": "usecases.ems.vpp_envs",
                  "rl.__init__": "usecases.setcover.rl.__init__",
                  "rl.algos": "usecases.setcover.rl.algos",
                  "rl.utility": "usecases.setcover.rl.utility"}

# Timesteps of 15 minutes in a day for the EMS use case
TIMESTEP_IN_A_DAY = 96

# Supported training methods for the EMS use case
METHODS = ['unify-all-at-once', 'unify-sequential', 'rl-all-at-once', 'rl-sequential']

# Training or inference mode
MODES = ['train', 'test']

# Reward shaping due to constraints violation
MIN_REWARD = -10000
PYTHON_VERSION = sys.version_info

# Due to the serialization procedure of the garage library, the version of pickle we choose depends on the Python
# version
assert PYTHON_VERSION.major == 3, "Only Python 3.7 or 3.8 are supported"
assert PYTHON_VERSION.minor in {7, 8}, "Only Python 3.7 or 3.8 are supported"

if PYTHON_VERSION.minor == 7:
    import pickle5 as pickle
else:
    import pickle

########################################################################################################################


class RenameUnpickler(pickle.Unpickler):
    """
    Custom Unpickler to load pkl object with old and legacy modules.
    """

    def find_class(self, module, name):
        """
        Override the find_class method of pickle, replacing legacy module names with new ones.
        :param module: str; module name.
        :param name: str; class name.
        :return:
        """
        renamed_module = module

        if module in LEGACY_MODULES.keys():
            renamed_module = LEGACY_MODULES[module]

        return super(RenameUnpickler, self).find_class(renamed_module, name)


########################################################################################################################


class CustomExperimentTemplate:
    """
    Custom implementation of garage.experiment.experiment.ExperimentTemplate to avoid environment serialization
    (https://github.com/rlworkgroup/garage/blob/3492f446633a7e748f2f79077f6301c5b3ec9281/src/garage/experiment/experiment.py)

    Creates experiment log directories and runs an experiment.

    This class should only be created by calling garage.wrap_experiment.
    Generally, it's used as a decorator like this:

        @wrap_experiment(snapshot_mode='all')
        def my_experiment(ctxt, seed, lr=0.5):
            ...

        my_experiment(seed=1)

    Even though this class could be implemented as a closure in
    wrap_experiment(), it's more readable (and easier to pickle) implemented as
    a class.

    Note that the full path that will be created is
    f'{data}/local/{prefix}/{name}'.

    Args:
        function (callable or None): The experiment function to wrap.
        log_dir (str or None): The full log directory to log to. Will be
            computed from `name` if omitted.
        name (str or None): The name of this experiment template. Will be
            filled from the wrapped function's name if omitted.
        prefix (str): Directory under data/local in which to place the
            experiment directory.
        snapshot_mode (str): Policy for which snapshots to keep (or make at
            all). Can be either "all" (all iterations will be saved), "last"
            (only the last iteration will be saved), "gap" (every snapshot_gap
            iterations are saved), or "none" (do not save snapshots).
        snapshot_gap (int): Gap between snapshot iterations. Waits this number
            of iterations before taking another snapshot.
        archive_launch_repo (bool): Whether to save an archive of the
            repository containing the launcher script. This is a potentially
            expensive operation which is useful for ensuring reproducibility.
        name_parameters (str or None): Parameters to insert into the experiment
            name. Should be either None (the default), 'all' (all parameters
            will be used), or 'passed' (only passed parameters will be used).
            The used parameters will be inserted in the order they appear in
            the function definition.
        use_existing_dir (bool): If true, (re)use the directory for this
            experiment, even if it already contains data.
        x_axis (str): Key to use for x axis of plots.



    """

    # pylint: disable=too-few-public-methods

    def __init__(self, *, function, log_dir, name, prefix, snapshot_mode,
                 snapshot_gap, archive_launch_repo, name_parameters,
                 use_existing_dir, x_axis):
        self.function = function
        self.log_dir = log_dir
        self.name = name
        self.prefix = prefix
        self.snapshot_mode = snapshot_mode
        self.snapshot_gap = snapshot_gap
        self.archive_launch_repo = archive_launch_repo
        self.name_parameters = name_parameters
        self.use_existing_dir = use_existing_dir
        self.x_axis = x_axis
        if self.function is not None:
            self._update_wrap_params()

    def _update_wrap_params(self):
        """Update self to "look like" the wrapped funciton.

        Mostly, this involves creating a function signature for the
        ExperimentTemplate that looks like the wrapped function, but with the
        first argument (ctxt) excluded, and all other arguments required to be
        keyword only.

        """
        functools.update_wrapper(self, self.function)
        self.__signature__ = _make_experiment_signature(self.function)

    @classmethod
    def _augment_name(cls, options, name, params):
        """Augment the experiment name with parameters.

        Args:
            options (dict): Options to `wrap_experiment` itself. See the
                function documentation for details.
            name (str): Name without parameter names.
            params (dict): Dictionary of parameters.

        Raises:
            ValueError: If self.name_parameters is not set to None, "passed",
                or "all".

        Returns:
            str: Returns the augmented name.

        """
        name_parameters = collections.OrderedDict()

        if options['name_parameters'] == 'passed':
            for param in options['signature'].parameters.values():
                try:
                    name_parameters[param.name] = params[param.name]
                except KeyError:
                    pass
        elif options['name_parameters'] == 'all':
            for param in options['signature'].parameters.values():
                name_parameters[param.name] = params.get(
                    param.name, param.default)
        elif options['name_parameters'] is not None:
            raise ValueError('wrap_experiment.name_parameters should be set '
                             'to one of None, "passed", or "all"')
        param_str = '_'.join('{}={}'.format(k, v)
                             for (k, v) in name_parameters.items())
        if param_str:
            return '{}_{}'.format(name, param_str)
        else:
            return name

    def _get_options(self, *args):
        """Get the options for wrap_experiment.

        This method combines options passed to `wrap_experiment` itself and to
        the wrapped experiment.

        Args:
            args (list[dict]): Unnamed arguments to the wrapped experiment. May
                be an empty list or a list containing a single dictionary.

        Raises:
            ValueError: If args contains more than one value, or the value is
                not a dictionary containing at most the same keys as are
                arguments to `wrap_experiment`.

        Returns:
            dict: The final options.

        """
        options = dict(name=self.name,
                       function=self.function,
                       prefix=self.prefix,
                       name_parameters=self.name_parameters,
                       log_dir=self.log_dir,
                       archive_launch_repo=self.archive_launch_repo,
                       snapshot_gap=self.snapshot_gap,
                       snapshot_mode=self.snapshot_mode,
                       use_existing_dir=self.use_existing_dir,
                       x_axis=self.x_axis,
                       signature=self.__signature__)
        if args:
            if len(args) == 1 and isinstance(args[0], dict):
                for k in args[0]:
                    if k not in options:
                        raise ValueError('Unknown key {} in wrap_experiment '
                                         'options'.format(k))
                options.update(args[0])
            else:
                raise ValueError('garage.experiment currently only supports '
                                 'keyword arguments')
        return options

    @classmethod
    def _make_context(cls, options, **kwargs):
        """Make a context from the template information and variant args.

        Currently, all arguments should be keyword arguments.

        Args:
            options (dict): Options to `wrap_experiment` itself. See the
                function documentation for details.
            kwargs (dict): Keyword arguments for the wrapped function. Will be
                logged to `variant.json`

        Returns:
            ExperimentContext: The created experiment context.

        """
        name = options['name']
        if name is None:
            name = options['function'].__name__
        name = cls._augment_name(options, name, kwargs)
        log_dir = options['log_dir']
        if log_dir is None:
            log_dir = ('{data}/local/{prefix}/{name}'.format(
                data=os.path.join(os.getcwd(), 'data'),
                prefix=options['prefix'],
                name=name))
        if options['use_existing_dir']:
            os.makedirs(log_dir, exist_ok=True)
        else:
            log_dir = _make_sequential_log_dir(log_dir)

        tabular_log_file = os.path.join(log_dir, 'progress.csv')
        text_log_file = os.path.join(log_dir, 'debug.log')
        variant_log_file = os.path.join(log_dir, 'variant.json')
        metadata_log_file = os.path.join(log_dir, 'metadata.json')

        kwargs_to_dump = kwargs.copy()
        # Since it may require a lot of memory, remove the environment from the serialization
        if 'env' in kwargs_to_dump.keys():
            del kwargs_to_dump['env']
        dump_json(variant_log_file, kwargs_to_dump)
        git_root_path, metadata = get_metadata()
        dump_json(metadata_log_file, metadata)
        if git_root_path and options['archive_launch_repo']:
            make_launcher_archive(git_root_path=git_root_path, log_dir=log_dir)

        logger.add_output(dowel.TextOutput(text_log_file))
        logger.add_output(dowel.CsvOutput(tabular_log_file))
        logger.add_output(
            dowel.TensorBoardOutput(log_dir, x_axis=options['x_axis']))
        logger.add_output(dowel.StdOutput())

        logger.push_prefix('[{}] '.format(name))
        logger.log('Logging to {}'.format(log_dir))

        return ExperimentContext(snapshot_dir=log_dir,
                                 snapshot_mode=options['snapshot_mode'],
                                 snapshot_gap=options['snapshot_gap'])

    def __call__(self, *args, **kwargs):
        """Wrap a function to turn it into an ExperimentTemplate.

        Note that this docstring will be overriden to match the function's
        docstring on the ExperimentTemplate once a function is passed in.

        Args:
            args (list): If no function has been set yet, must be a list
                containing a single callable. If the function has been set, may
                be a single value, a dictionary containing overrides for the
                original arguments to `wrap_experiment`.
            kwargs (dict): Arguments passed onto the wrapped function.

        Returns:
            object: The returned value of the wrapped function.

        Raises:
            ValueError: If not passed a single callable argument.

        """
        if self.function is None:
            if len(args) != 1 or len(kwargs) != 0 or not callable(args[0]):
                raise ValueError('Please apply the result of '
                                 'wrap_experiment() to a single function')
            # Apply ourselves as a decorator
            self.function = args[0]
            self._update_wrap_params()
            return self
        else:
            ctxt = self._make_context(self._get_options(*args), **kwargs)
            result = self.function(ctxt, **kwargs)
            logger.remove_all()
            logger.pop_prefix()
            gc.collect()  # See dowel issue #44
            return result

########################################################################################################################


def get_metadata():
    """
    This custom implementation fix a bug for Windows OS system.
    Get metadata about the main script.

    The goal of this function is to capture the additional information needed
    to re-run an experiment, assuming that the launcher script that started the
    experiment is located in a clean git repository.

    Returns:
        tuple[str, dict[str, str]]:
          * Absolute path to root directory of launcher's git repo.
          * Directory containing:
            * githash (str): Hash of the git revision of the repo the
                experiment was started from. "-dirty" will be appended to this
                string if the repo has uncommitted changes. May not be present
                if the main script is not in a git repo.
            * launcher (str): Relative path to the main script from the base of
                the repo the experiment was started from. If the main script
                was not started from a git repo, this will instead be an
                absolute path to the main script.

    """
    main_file = getattr(main, '__file__', None)
    if not main_file:
        return None, {}
    main_file_path = os.path.abspath(main_file)
    try:
        git_root_path = subprocess.check_output(
            ('git', 'rev-parse', '--show-toplevel'),
            cwd=os.path.dirname(main_file_path),
            stderr=subprocess.DEVNULL)
        git_root_path = git_root_path.strip()
    except subprocess.CalledProcessError:
        # This file is always considered not to exist.
        git_root_path = ''
    # We check that the path exists since in old versions of git the above
    # rev-parse command silently exits with 0 when run outside of a git repo.
    if not os.path.exists(git_root_path):
        return None, {
            'launcher': main_file_path,
        }
    launcher_path = os.path.relpath(bytes(main_file_path, encoding='utf8'),
                                    git_root_path)

    # NOTE: fixed bug for Windows OS
    if os.name == 'nt':
        git_root_path = git_root_path.decode("utf-8")

    git_hash = subprocess.check_output(('git', 'rev-parse', 'HEAD'),
                                       cwd=git_root_path)
    git_hash = git_hash.decode('utf-8').strip()
    git_status = subprocess.check_output(('git', 'status', '--short'),
                                         cwd=git_root_path)
    git_status = git_status.decode('utf-8').strip()
    if git_status != '':
        git_hash = git_hash + '-dirty'
    return git_root_path, {
        'githash': git_hash,
        'launcher': launcher_path.decode('utf-8'),
    }

########################################################################################################################


def renamed_load(file_obj):
    """
    This method is the same as pickle.load().
    :param file_obj: binary file.
    :return: de-serialized object.
    """
    return RenameUnpickler(file_obj).load()


def renamed_loads(pickled_bytes):
    """
    This method is the same as pickle.loads().
    :param pickled_bytes: bytes to unpickle.
    :return: de-serialized object.
    """
    file_obj = io.BytesIO(pickled_bytes)
    return renamed_load(file_obj)

########################################################################################################################


def my_wrap_experiment(function,
                       logging_dir,
                       snapshot_mode,
                       snapshot_gap,
                       *,
                       prefix='experiment',
                       name=None,
                       archive_launch_repo=True,
                       name_parameters=None,
                       use_existing_dir=True,
                       x_axis='TotalEnvSteps'):
    """
    Custom wrapper for the ExperimentTemplate class of the garage library that allows to set the log directory.
    See the ExperimentTemplate class for more details.
    """
    return CustomExperimentTemplate(function=function,
                                    log_dir=logging_dir,
                                    prefix=prefix,
                                    name=name,
                                    snapshot_mode=snapshot_mode,
                                    snapshot_gap=snapshot_gap,
                                    archive_launch_repo=archive_launch_repo,
                                    name_parameters=name_parameters,
                                    use_existing_dir=use_existing_dir,
                                    x_axis=x_axis)

########################################################################################################################


def min_max_scaler(starting_range: Tuple[Union[float, int]],
                   new_range: Tuple[Union[float, int]],
                   value: float) -> float:
    """
    Scale the input value from a starting range to a new one.
    :param starting_range: tuple of float; the starting range.
    :param new_range: tuple of float; the new range.
    :param value: float; value to be rescaled.
    :return: float; rescaled value.
    """

    assert isinstance(starting_range, tuple) and len(starting_range) == 2, \
        "feature_range must be a tuple as (min, max)"
    assert isinstance(new_range, tuple) and len(new_range) == 2, \
        "feature_range must be a tuple as (min, max)"

    min_start_value = starting_range[0]
    max_start_value = starting_range[1]
    min_new_value = new_range[0]
    max_new_value = new_range[1]

    value_std = (value - min_start_value) / (max_start_value - min_start_value)
    scaled_value = value_std * (max_new_value - min_new_value) + min_new_value

    return scaled_value

########################################################################################################################


def timestamps_headers(num_timeunits: int) -> List[str]:
    """
    Given a number of timeunits (in minutes), it provides a string representation of each timeunit.
    For example, if num_timeunits=96, the result is [00:00, 00:15, 00:30, ...].
    :param num_timeunits: int; the number of timeunits in a day.
    :return: list of string; list of timeunits.
    """

    start_time = datetime.strptime('00:00', '%H:%M')
    timeunit = 24 * 60 / num_timeunits
    timestamps = [start_time + idx * timedelta(minutes=timeunit) for idx in range(num_timeunits)]
    timestamps = ['{:02d}:{:02d}'.format(timestamp.hour, timestamp.minute) for timestamp in timestamps]

    return timestamps

########################################################################################################################


def instances_preprocessing(instances: pd.DataFrame) -> pd.DataFrame:
    """
    Convert PV and Load values from string to float.
    :param instances: pandas.Dataframe; PV and Load for each timestep and for every instance.
    :return: pandas.Dataframe; the same as the input dataframe but with float values instead of string.
    """

    assert 'PV(kW)' in instances.keys(), "PV(kW) must be in the dataframe columns"
    assert 'Load(kW)' in instances.keys(), "Load(kW) must be in the dataframe columns"

    # Instances pv from file
    instances['PV(kW)'] = instances['PV(kW)'].map(lambda entry: entry[1:-1].split())
    instances['PV(kW)'] = instances['PV(kW)'].map(lambda entry: list(np.float_(entry)))

    # Instances load from file
    instances['Load(kW)'] = instances['Load(kW)'].map(lambda entry: entry[1:-1].split())
    instances['Load(kW)'] = instances['Load(kW)'].map(lambda entry: list(np.float_(entry)))

    return instances


