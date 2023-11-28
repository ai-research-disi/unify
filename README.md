# UNIFY: a Unified Policy Designing Framework for Solving Integrated Constrained Optimization and Machine Learning Problems

Code to reproduce results of the paper *UNIFY: a Unified Policy Designing Framework for Solving Constrained Optimization 
Problems with Machine Learning* under review at AAAI-2023 conference.

## Repository description

The repository is structured as follows:

- `data/ems`. Directory with all the required data to reproduce experiments regarding the Energy Management 
System (EMS), namely:
  - `Dataset10k.csv`: photovoltaic and user demands forecast.
  - `gmePrices.npy`: energy prices.
  - `instance_indexes.csv`: indexes of the instances used for the evaluation on the EMS.
  - `optShift.npy`: optimal power shifts; you can consider them as fixed parameters.
- `helpers`.
  - `online_heuristic.py`: implementation of the simple greedy heuristic provided with the paper *Hybrid Offline/Online 
  Optimization for Energy Management via Reinforcement Learning*.
  - `utility.py`: utility functions shared across the use cases.
- `notebooks`.   
  - `energy_management_system.ipynb`: notebook to reproduce step-by-step experiments on the EMS.
  - `set_multi_cover.ipynb`: notebook to reproduce step-by-step experiments on the Set Multi-cover.
- `test`.
  - `test_msc.py`: unittest for the Set Multi-cover experiments.
- `usecases`.
  - `ems`.
    - `rl`.
      - `algos.py`: custom implementation of the Vanilla Policy Gradient from `garage` to keep track of the true 
      solution cost.
      - `utility.py`: custom implementation of the training and logging routine of `garage` to keep track of the true 
      solution cost.
    - `plotting.py`: methods to visualize results.
    - `train_and_test_rl.py`: main execution of the training and test routines.
    - `vpp_envs.py`: `gym` environments for this use case.
  - `setcover`
    - `rl`.
      - `algos.py`: custom implementation of the Vanilla Policy Gradient from `garage` to keep track of the solution 
      cost.
      - `utility.py`: custom implementation of the training and logging routine of `garage` to keep track of the  
      solution cost. 
    - `generate_instances.py`: methods to generate Set Multi-cover instances.
    - `plotting.py`: methods to visualize the results.
    - `predict_then_optimize.py`: methods to train and evaluate the Predict-then-Optimize approach.
    - `rl_train_and_test.py`: main execution of the training and test routines.
    - `solve_instances.py`: methods to solve the Set Multi-cover instances.
    - `stochastic_algorithm.py`: implementation utilities for the stochastic algorithm.
- `launch_exp.sh`: Linux bash file to run all the experiments at once.

## Run the experiments

You can choose one of the following options to run the experiments:
 - **Notebooks**. Run the notebooks in the `notebooks` folder.
 - **Python scripts**. You can take a look at the Python scripts/modules to run only what you need:
   - `python -m ems`. 
      Training and test routines for the EMS use case. 
      Please, use the `--help` option for the full description. 
      To reproduce experiments from the **Offline/Online Integration using UNIFY** paragraph of the paper, you have to 
      choose `unify-sequential` and `unify-all-at-once` methods.
      To reproduce experiments from the **Constrained RL** paragraph of the paper, you have to 
      choose `rl-sequential` with and without the `--safety-layer` flag.
   - `python -m setcover`: training and test routines for the EMS use case; please use the `--help` option for the full 
   description.
   - `python usecases/setcover/predict_then_optimize.py`: training and test of the Predict-then-optimize approach; you 
   must run the script from the main project directory.
   - `python usecases/setcover/stochastic_algorithm.py`: run the stochastic algorithm; you must run the script from the main project directory.
 - **launch_exp.sh**: Linux bash script to run the experiments all at once.
 
 This code was tested on Python 3.7 and 3.8, and Windows 10 OS and CentoOS 8.