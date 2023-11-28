"""
    Plotting functions.
"""

import os
import pickle
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from usecases.setcover.generate_instances import load_msc
from usecases.setcover.stochastic_algorithm import evaluate_stochastic_algo

#sns.set_style('darkgrid')

########################################################################################################################


def plot_evaluation_wrt_num_scenarios(data_path,
                                      stochastic_algo_res_dir,
                                      rl_res_dir,
                                      num_scenarios,
                                      seed,
                                      test_split,
                                      title=None):
    """
    Plot the cost wrt the number of scenarios for DFL approach and stochastic algorithm.
    :param data_path: str; where the data instances are loaded from.
    :param stochastic_algo_res_dir: str; where the stochastic algorithm results are loaded from (pickle format).
    :param rl_res_dir: str; where the RL results are loaded from (pickle format).
    :param num_scenarios: list of int; the number of scenarios employed for the stochastic algorithm.
    :param seed: int; the seed used during instances generation and training.
    :param test_split: float; split between training and test.
    :param title: str; the plot title.
    :return:
    """

    data_path = os.path.join(data_path, f'seed-{seed}')
    stochastic_algo_res_dir = os.path.join(stochastic_algo_res_dir, f'seed-{seed}')
    rl_res_dir = os.path.join(rl_res_dir, f'seed-{seed}')

    instances = list()
    optimal_costs = list()
    stochastic_algo_training_costs = list()
    stochastic_algo_test_costs = list()

    print('Loading instances...')
    for f in os.listdir(data_path):
        instances.append(load_msc(os.path.join(data_path, f, 'instance.pkl')))
        optimal_cost = pickle.load(open(os.path.join(data_path, f, 'optimal-cost.pkl'), 'rb'))
        optimal_costs.append(optimal_cost)
    print('Finished')

    # Split between training and test instances
    train_instances, test_instances, \
        train_optimal_costs, test_optimal_costs = \
            train_test_split(instances, optimal_costs, test_size=test_split, random_state=seed)

    # Compute the stochastic algorithm cost for training and test sets
    for num in num_scenarios:
        stochastic_algo_res = \
            evaluate_stochastic_algo(res=stochastic_algo_res_dir,
                                     num_scenarios=num,
                                     instances=train_instances,
                                     optimal_costs=train_optimal_costs)

        stochastic_algo_training_costs.append(stochastic_algo_res['Stochastic algo mean cost'])
        train_mean_optimal_cost = stochastic_algo_res['Mean optimal cost']


        stochastic_algo_res = \
            evaluate_stochastic_algo(res=stochastic_algo_res_dir,
                                     num_scenarios=num,
                                     instances=test_instances,
                                     optimal_costs=test_optimal_costs)

        stochastic_algo_test_costs.append(stochastic_algo_res['Stochastic algo mean cost'])
        test_mean_optimal_cost = stochastic_algo_res['Mean optimal cost']

    # Load the RL cost on training and test sets
    progress = pd.read_csv(os.path.join(rl_res_dir, 'progress.csv'))
    rl_training_cost = progress['Evaluation/TrainBenchmarkAverageCost']
    rl_training_sampled_instances = np.arange(len(rl_training_cost)) * 100
    rl_test_cost = progress['Evaluation/TestBenchmarkAverageCost']
    rl_test_sampled_instances = np.arange(len(rl_test_cost)) * 100

    # Find the minimum cost on training and test instances for visualization purposes
    min_rl_training_cost = np.min(rl_training_cost)
    min_rl_training_cost_idx = np.argmin(rl_training_cost)
    min_rl_training_cost_scenario = rl_training_sampled_instances[min_rl_training_cost_idx]
    min_stochastic_training_cost = np.min(stochastic_algo_training_costs)
    min_stochastic_training_cost_idx = np.argmin(stochastic_algo_training_costs)
    min_stochastic_training_cost_scenario = num_scenarios[min_stochastic_training_cost_idx]
    best_training_costs = [min_rl_training_cost, min_stochastic_training_cost]
    best_training_costs_scenarios = [min_rl_training_cost_scenario, min_stochastic_training_cost_scenario]
    best_training_cost, training_idx = min(best_training_costs), np.argmin(best_training_costs)
    best_training_cost_scenario = best_training_costs_scenarios[training_idx]

    min_rl_test_cost = np.min(rl_test_cost)
    min_rl_test_cost_idx = np.argmin(rl_test_cost)
    min_rl_test_cost_scenario = rl_test_sampled_instances[min_rl_test_cost_idx]
    min_stochastic_test_cost = np.min(stochastic_algo_test_costs)
    min_stochastic_test_cost_idx = np.argmin(stochastic_algo_test_costs)
    min_stochastic_test_cost_scenario = num_scenarios[min_stochastic_test_cost_idx]
    best_test_costs = [min_rl_test_cost, min_stochastic_test_cost]
    best_test_costs_scenarios = [min_rl_test_cost_scenario, min_stochastic_test_cost_scenario]
    best_test_cost, test_idx = min(best_test_costs), np.argmin(best_test_costs)
    best_test_cost_scenario = best_test_costs_scenarios[test_idx]

    print(f'Min RL cost: {min_rl_training_cost} | Min stochastic cost: {min_stochastic_training_cost}')

    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(10, 7))
    plt.subplots_adjust(wspace=0.05)
    ax1.plot(rl_training_sampled_instances, rl_training_cost, label='RL')
    ax1.plot(num_scenarios, stochastic_algo_training_costs, label='Stochastic algorithm on training')
    ax1.axhline(train_mean_optimal_cost, linestyle='--', color='r', label='Optimal cost on training')
    ax1.scatter(best_training_cost_scenario, best_training_cost, marker="*", color='y', label='Best cost found')
    ax1.set_xlabel('# of scenarios')
    ax1.set_title('Training', fontsize=12, fontweight='bold')

    ax2.plot(rl_test_sampled_instances, rl_test_cost, label='RL')
    ax2.plot(num_scenarios, stochastic_algo_test_costs, label='Stochastic algorithm')
    ax2.axhline(test_mean_optimal_cost, linestyle='--', color='r', label='Optimal cost')
    ax2.scatter(best_test_cost_scenario, best_test_cost, marker="*", color='y', label='Best cost found')
    ax2.set_xlabel('# of scenarios')
    ax2.set_title('Test', fontsize=12, fontweight='bold')
    ax1.legend()
    ax2.legend()
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.show()

########################################################################################################################


def plot_rl_stochastic_algo_evaluation(data_path,
                                       stochastic_algo_res_dir,
                                       rl_res_dir,
                                       seed,
                                       test_split,
                                       num_instances,
                                       title=None):
    """
    Plot the cost wrt the number of scenarios for DFL approach and stochastic algorithm.
    :param data_path: str; where the data instances are loaded from.
    :param stochastic_algo_res_dir: str; where the stochastic algorithm results are loaded from (pickle format).
    :param rl_res_dir: str; where the RL results are loaded from (pickle format).
    :param seed: int; the seed used during instances generation and training.
    :param test_split: float; split between training and test.
    :param num_instances: int; number instances.
    :param title: str; the plot title.
    :return:
    """

    data_path = os.path.join(data_path, f'{num_instances}-instances', f'seed-{seed}')
    stochastic_algo_res_dir = os.path.join(stochastic_algo_res_dir, f'{num_instances}-instances', f'seed-{seed}')
    rl_res_dir = os.path.join(rl_res_dir, f'{num_instances}-instances', f'seed-{seed}')

    instances = list()
    optimal_costs = list()

    print('Loading instances...')
    for f in os.listdir(data_path):
        instances.append(load_msc(os.path.join(data_path, f, 'instance.pkl')))
        optimal_cost = pickle.load(open(os.path.join(data_path, f, 'optimal-cost.pkl'), 'rb'))
        optimal_costs.append(optimal_cost)
    print('Finished')

    # Split between training and test instances
    train_instances, test_instances, \
        train_optimal_costs, test_optimal_costs = \
            train_test_split(instances, optimal_costs, test_size=test_split, random_state=seed)

    # The best results are achieved with a number of scenarios equal the training set size
    num_scenarios = len(train_instances)
    print(num_scenarios)

    # Compute the stochastic algorithm cost for training and test sets
    train_res = \
        evaluate_stochastic_algo(res=stochastic_algo_res_dir,
                                 num_scenarios=num_scenarios,
                                 instances=train_instances,
                                 optimal_costs=train_optimal_costs)

    test_res = \
        evaluate_stochastic_algo(res=stochastic_algo_res_dir,
                                 num_scenarios=num_scenarios,
                                 instances=test_instances,
                                 optimal_costs=test_optimal_costs)

    # Load RL results
    rl_train_res = pickle.load(open(os.path.join(rl_res_dir, 'train-set-res.pkl'), 'rb'))
    rl_test_res = pickle.load(open(os.path.join(rl_res_dir, 'test-set-res.pkl'), 'rb'))

    # Print mean cost on training and test sets
    print('TRAINING SET')
    print(f'Mean optimal cost: {train_res["Mean optimal cost"]}' + \
          f' | Mean stochastic algorithm cost: {train_res["Stochastic algo mean cost"]}' + \
          f' | Mean RL cost: {rl_train_res["Mean cost"]}')

    print('TEST SET')
    print(f'Mean optimal cost: {test_res["Mean optimal cost"]}' + \
          f'| Mean stochastic algorithm cost: {test_res["Stochastic algo mean cost"]}' +
          f' | Mean RL cost: {rl_test_res["Mean cost"]}')

    # Visualize results in a bar plot
    columns = ['Optimal', 'RL', 'Stochastic algo.']
    x_pos = np.arange(len(columns))

    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(10, 7))
    plt.subplots_adjust(wspace=0.1)

    cost_values = [train_res["Mean optimal cost"],
                   rl_train_res['Mean cost'],
                   train_res["Stochastic algo mean cost"]]
    std_cost_values = [train_res["Std optimal cost"],
                       rl_train_res['Std dev'],
                       train_res["Stochastic algo std cost"]]

    ax1.bar(x_pos,
            cost_values,
            yerr=std_cost_values,
            align='center',
            alpha=1)
    ax1.set_xticks(x_pos)
    ax1.set_yticks(cost_values)
    ax1.set_xticklabels(columns)
    ax1.yaxis.grid(True)
    ax1.set_title('Training set')

    cost_values = [test_res["Mean optimal cost"],
                   rl_test_res['Mean cost'],
                   test_res["Stochastic algo mean cost"]]
    std_cost_values = [test_res["Std optimal cost"],
                       rl_test_res['Std dev'],
                       test_res["Stochastic algo std cost"]]
    ax2.bar(x_pos,
            cost_values,
            align='center',
            alpha=1,
            yerr=std_cost_values)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(columns)
    ax2.set_yticks(cost_values)
    #ax2.yaxis.grid(True)
    ax2.set_title('Test set')

    fig.suptitle('Mean cost', fontsize=14, fontweight='bold')

    plt.show()

########################################################################################################################


def plot_rl_pto_evaluation(data_path,
                           pto_res_dir,
                           rl_res_dir,
                           seed,
                           test_split,
                           num_instances,
                           acronym):
    """
    Plot the cost wrt the number of scenarios for DFL approach and stochastic algorithm.
    :param data_path: str; where the data instances are loaded from.
    :param pto_res_dir: str; where the predict-then-optimize algorithm results are loaded from (pickle format).
    :param rl_res_dir: str; where the RL results are loaded from (pickle format).
    :param seed: int; the seed used during instances generation and training.
    :param test_split: float; split between training and test.
    :param num_instances: int; number instances.
    :param title: str; the plot title.
    :return:
    """

    data_path = os.path.join(data_path, f'{num_instances}-instances', f'seed-{seed}')
    pto_res_dir = os.path.join(pto_res_dir, f'{num_instances}-instances', 'single-scenario', f'seed-{seed}')
    rl_res_dir = os.path.join(rl_res_dir, f'{num_instances}-instances', f'seed-{seed}')

    instances = list()
    optimal_costs = list()

    print('Loading instances...')
    for f in os.listdir(data_path):
        instances.append(load_msc(os.path.join(data_path, f, 'instance.pkl')))
        optimal_cost = pickle.load(open(os.path.join(data_path, f, 'optimal-cost.pkl'), 'rb'))
        optimal_costs.append(optimal_cost)
    print('Finished')

    # Split between training and test instances
    _, test_optimal_costs = \
            train_test_split(optimal_costs, test_size=test_split, random_state=seed)

    # Load RL results
    rl_test_res = pickle.load(open(os.path.join(rl_res_dir, 'test-set-res.pkl'), 'rb'))

    # Load predict-then-optimize results
    pto_res = pickle.load(open(os.path.join(pto_res_dir, 'costs.pkl'), 'rb'))
    assert 1 in pto_res.keys(), "Results for single scenario are missing"
    costs = pto_res[1]

    # Compute mean and std dev for predict-then-optimize
    pto_mean_cost = np.mean(costs)
    pto_std_cost = np.std(costs)

    # Compute mean and std dev for the optimal costs
    mean_optimal_cost = np.mean(test_optimal_costs)
    std_optimal_cost = np.std(test_optimal_costs)

    print('TEST SET')
    print(f'Mean optimal cost: {mean_optimal_cost}' + \
          f' | Mean RL cost: {rl_test_res["Mean cost"]}' + \
          f' | Mean Predict-then-optimize: {pto_mean_cost}')

    # Visualize results in a bar plot
    columns = ['Optimal', acronym, 'Predict-then-optimize']
    x_pos = np.arange(len(columns))

    cost_values = [mean_optimal_cost,
                   rl_test_res['Mean cost'],
                   pto_mean_cost]
    std_cost_values = [std_optimal_cost,
                       rl_test_res['Std dev'],
                       pto_std_cost]
    '''plt.bar(x_pos,
            cost_values,
            align='center',
            alpha=1,
            yerr=std_cost_values,
            cmap='virdis')

    plt.xticks(x_pos)
    plt.gca().set_xticklabels(columns)
    plt.yticks([cost_values[0], cost_values[-1]])'''
    #ax2.yaxis.grid(True)

    cost_values = np.expand_dims(cost_values, axis=0)
    df = pd.DataFrame(data=cost_values, columns=columns)
    sns.barplot(data=df, palette='Greens', yerr=std_cost_values)
    # plt.title('Mean cost', fontweight='bold', fontsize=14)
    plt.savefig('dfl-generalization-2.png', dpi=1000)

    plt.show()

########################################################################################################################


def plot_pto_evaluation(data_path,
                        stochastic_algo_res_dir,
                        pto_res_dir,
                        rl_res_dir,
                        seed,
                        num_instances,
                        test_split,
                        acronym):

    # Keep track of results collected with different seeds
    experiments_rl_mean_costs = list()
    experiments_pto_mean_runtimes = list()
    experiments_pto_mean_costs = list()
    experiments_stoc_algo_mean_costs = list()

    for s in seed:

        # Data and results directory
        current_data_path = os.path.join(data_path, f'{num_instances}-instances', f'seed-{s}')
        current_stochastic_algo_res_dir = os.path.join(stochastic_algo_res_dir, f'{num_instances}-instances', f'seed-{s}')
        current_pto_res_dir = os.path.join(pto_res_dir, f'{num_instances}-instances', f'seed-{s}')
        current_rl_res_dir = os.path.join(rl_res_dir, f'{num_instances}-instances', f'seed-{s}')

        instances = list()
        optimal_costs = list()

        # Load instances from file
        print('Loading instances...')
        for f in os.listdir(current_data_path):
            instances.append(load_msc(os.path.join(current_data_path, f, 'instance.pkl')))
            optimal_cost = pickle.load(open(os.path.join(current_data_path, f, 'optimal-cost.pkl'), 'rb'))
            optimal_costs.append(optimal_cost)
        print('Finished')

        # Split between training and test instances
        train_instances, test_instances, \
        train_optimal_costs, test_optimal_costs = \
            train_test_split(instances, optimal_costs, test_size=test_split, random_state=s)

        # Load Predict-then-optimize results
        costs = pickle.load(open(os.path.join(current_pto_res_dir, 'costs.pkl'), 'rb'))
        runtimes = pickle.load(open(os.path.join(current_pto_res_dir, 'runtimes.pkl'), 'rb'))

        # Consider only the number of scenarios and instances used to evaluate Predict-then-optimize
        assert costs.keys() == runtimes.keys()
        key_0 = list(costs.keys())[0]
        true_evaluated_instances = len(costs[key_0])
        evaluated_num_scenarios = list(costs.keys())

        for cost_key, runtimes_key in zip(costs.keys(), runtimes.keys()):
            assert cost_key == runtimes_key
            assert len(costs[cost_key]) == len(runtimes[runtimes_key])

            assert len(costs[cost_key]) == true_evaluated_instances
            assert len(runtimes[cost_key]) == true_evaluated_instances

        # Mean optimal cost on test instances
        mean_optimal_cost = np.mean(test_optimal_costs[:true_evaluated_instances])

        # Load RL results
        rl_res = pickle.load(open(os.path.join(current_rl_res_dir, 'test-set-res.pkl'), 'rb'))
        rl_mean_cost = rl_res['Mean cost']
        rl_mean_runtime = rl_res['Mean runtime']
        rl_mean_cost = rl_mean_cost / mean_optimal_cost

        # Evaluate the stochastic algorithm only for the considered number of instances and scenarios
        all_pto_mean_costs = list()
        all_pto_std_costs = list()
        all_pto_mean_runtimes = list()
        all_pto_std_runtimes = list()
        all_stochastic_algo_mean_costs = list()
        all_stochastic_algo_std_costs = list()

        for num_scenarios in evaluated_num_scenarios:
            pto_mean_cost = np.mean(costs[num_scenarios])
            pto_std_cost = np.std(costs[num_scenarios])
            pto_mean_runtime = np.mean(runtimes[num_scenarios])
            pto_mean_runtime = pto_mean_runtime / rl_mean_runtime
            pto_std_runtime = np.std(runtimes[num_scenarios])

            stochastic_algo_res = evaluate_stochastic_algo(res=current_stochastic_algo_res_dir,
                                                           instances=test_instances[:true_evaluated_instances],
                                                           optimal_costs=test_optimal_costs[:true_evaluated_instances],
                                                           num_scenarios=num_scenarios)

            pto_mean_cost = pto_mean_cost / mean_optimal_cost
            stoc_algo_mean_cost = stochastic_algo_res['Stochastic algo mean cost'] / mean_optimal_cost

            all_pto_mean_costs.append(pto_mean_cost)
            all_pto_std_costs.append(pto_std_cost)
            all_pto_mean_runtimes.append(pto_mean_runtime)
            all_pto_std_runtimes.append(pto_std_runtime)
            all_stochastic_algo_mean_costs.append(stoc_algo_mean_cost)
            all_stochastic_algo_std_costs.append(stochastic_algo_res['Stochastic algo std cost'])

            print(f'Predict-then-optimize mean cost: {pto_mean_cost} | ' + \
                  f'Predict-then-optimize mean runtime: {pto_mean_runtime}')
            print(f'Stochastic algorithm mean cost: {stoc_algo_mean_cost}')
            print(f'RL mean cost: {rl_mean_cost}')

        experiments_pto_mean_costs.append(all_pto_mean_costs)
        experiments_rl_mean_costs.append(rl_mean_cost)
        experiments_stoc_algo_mean_costs.append(all_stochastic_algo_mean_costs)
        experiments_pto_mean_runtimes.append(all_pto_mean_runtimes)

    overall_pto_mean = np.mean(experiments_pto_mean_costs, axis=0)
    overall_stoc_algo_mean = np.mean(experiments_stoc_algo_mean_costs, axis=0)
    overall_rl_mean = np.mean(experiments_rl_mean_costs)
    overall_rl_mean = np.tile(overall_rl_mean, len(evaluated_num_scenarios))
    overall_pto_std = np.std(experiments_pto_mean_costs, axis=0)
    overall_rl_std = np.std(experiments_rl_mean_costs)
    overall_rl_std = np.tile(overall_rl_std, len(evaluated_num_scenarios))
    overall_stoc_algo_std = np.std(experiments_stoc_algo_mean_costs, axis=0)
    overall_pto_mean_runtime = np.mean(experiments_pto_mean_runtimes, axis=0)

    # Find when the predict-then-optimize surpasses the RL-DFL
    border_indexes = np.argwhere(overall_pto_mean < overall_rl_mean)
    border_indexes = np.squeeze(border_indexes)
    border_index = min(border_indexes)
    border_num_scenarios = evaluated_num_scenarios[border_index]

    # Plot results
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(7, 5))
    ax1.plot(evaluated_num_scenarios,
             overall_pto_mean,
             label='Predict-then-optimize',
             color='darkorange',
             linestyle=':')
    '''ax1.fill_between(evaluated_num_scenarios,
                     overall_pto_mean - overall_pto_std,
                     overall_pto_mean + overall_pto_std,
                     alpha=0.25,
                     color='orange')'''
    ax1.plot(evaluated_num_scenarios,
             overall_stoc_algo_mean,
             label='Stochastic algorithm',
             color='red',
             linestyle='-.')
    '''ax1.fill_between(evaluated_num_scenarios,
                     overall_stoc_algo_mean - overall_stoc_algo_std,
                     overall_stoc_algo_mean + overall_stoc_algo_std,
                     alpha=0.25,
                     color='red')'''
    ax1.plot(evaluated_num_scenarios,
             overall_rl_mean,
             label=acronym,
             color='green',
             linestyle='solid')
    '''ax1.fill_between(evaluated_num_scenarios,
                     overall_rl_mean - overall_rl_std,
                     overall_rl_mean + overall_rl_std,
                     alpha=0.25,
                     color='green')'''
    ax1.axvline(border_num_scenarios, linestyle='--', color='darkblue')
    ax1.set_title('Normalized cost', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax2.plot(evaluated_num_scenarios,
             overall_pto_mean_runtime,
             label='Predict-then-optimize',
             linestyle=':',
             color='darkorange')
    ax2.axvline(border_num_scenarios, linestyle='--', color='darkblue')
    ax2.set_title('Normalized runtime', fontsize=13, fontweight='bold')
    ax2.set_xlabel('# scenarios', fontsize=12)
    ax2.legend(fontsize=11)
    plt.savefig('replacing-sampling-1.png', dpi=1000)
    plt.show()

########################################################################################################################


def plot_mapes(loadpath,
               num_instances,
               seed):
    mapes = np.load(os.path.join(loadpath, f'{num_instances}-instances', f'seed-{seed}', 'mapes.npy'))
    lambdas = np.arange(len(mapes))
    plt.rcParams["figure.figsize"] = (10, 5)
    plt.bar(lambdas, mapes, color='g')
    plt.gca().set_xticklabels([])
    plt.gca().set_xticks([])
    # math text
    plt.xlabel(r'$\lambda$', fontsize=16)
    plt.title('Mean Absolute Percentage Error', fontweight='bold', fontsize=14)
    plt.show()

########################################################################################################################


if __name__ == '__main__':

    NUM_INSTANCES = 1000

    plot_mapes(loadpath='results/set-cover/stochastic/poisson/train-test-split/predict-then-optimize/200x1000/linear/',
               num_instances=NUM_INSTANCES,
               seed=0)

    plot_rl_pto_evaluation(data_path='data/set-cover/200x1000/linear/',
                           pto_res_dir='results/set-cover/stochastic/poisson/train-test-split/predict-then-optimize/200x1000/linear/',
                           rl_res_dir='results/set-cover/stochastic/poisson/train-test-split/rl/200x1000/linear/',
                           num_instances=NUM_INSTANCES,
                           seed=0,
                           test_split=0.5,
                           acronym='unify')

    plot_pto_evaluation(data_path='data/set-cover/200x1000/linear/',
                        stochastic_algo_res_dir='results/set-cover/stochastic/poisson/train-test-split/stochastic-algo/mean/200x1000/linear/',
                        pto_res_dir='results/set-cover/stochastic/poisson/train-test-split/predict-then-optimize/200x1000/linear/',
                        rl_res_dir='results/set-cover/stochastic/poisson/train-test-split/rl/200x1000/linear/',
                        seed=[1, 2, 3, 4],
                        num_instances=NUM_INSTANCES,
                        test_split=0.5,
                        acronym='unify')

