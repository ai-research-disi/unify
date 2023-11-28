#! /bin/bash
python -m usecases.ems results/models/pure-rl/mdp-env --method rl-sequential --epochs 52 --batch-size 9600 --mode train
python -m usecases.ems results/models/pure-rl/mdp-env --method rl-sequential --epochs 52 --batch-size 9600 --mode test
python -m usecases.ems results/models/rl-safety-layer/mdp-env --method rl-sequential --epochs 19 --batch-size 9600 --mode train --safety
python -m usecases.ems results/models/rl-safety-layer/mdp-env --method rl-sequential --epochs 19 --batch-size 9600 --mode test --safety
python -m usecases.ems results/models/hybrid-rl-opt/single-step/single-step-env --method unify-all-at-once --epochs 37 --batch-size 100 --mode train
python -m usecases.ems results/models/hybrid-rl-opt/single-step/single-step-env --method unify-all-at-once --epochs 37 --batch-size 100 --mode test
python -m usecases.ems results/models/hybrid-rl-opt/sequential/mdp-env --method unify-sequential --epochs 19 --batch-size 9600 --mode train
python -m usecases.ems results/models/hybrid-rl-opt/sequential/mdp-env --method unify-sequential --epochs 19 --batch-size 9600 --mode test
NUM_PRODS=200
NUM_SETS=1000
NUM_INSTANCES=1000
NUM_EVALUATED_INSTANCES=50
for SEED in {0..5}
  do
    python usecases/setcover/generate_instances.py --min-lambda 1 --max-lambda 10 --num-prods $NUM_PRODS --num-sets $NUM_SETS --density 0.02 --num-instances $NUM_INSTANCES --seed $SEED --evaluated-instances $NUM_EVALUATED_INSTANCES
    python --num-prods $NUM_PRODS --num-sets $NUM_SETS --batch-size 100 --num-epochs 10000 --seed $SEED --num-instances $NUM_INSTANCES --evaluated-instances $NUM_EVALUATED_INSTANCES --train
    python --num-prods $NUM_PRODS --num-sets $NUM_SETS --batch-size 100 --num-epochs 10000 --seed $SEED --num-instances $NUM_INSTANCES --evaluated-instances $NUM_EVALUATED_INSTANCES
    python usecases/setcover/stochastic_algorithm.py --num-prods $NUM_PRODS --num-sets $NUM_SETS --num-scenarios 1 10 20 30 50 75 100 --test-split 0.5 --seed $SEED --num-instances $NUM_INSTANCES --evaluated-instances $NUM_EVALUATED_INSTANCES
    python usecases/setcover/predict_then_optimize.py --seed $SEED --num-prods $NUM_PRODS --num-sets $NUM_SETS --num-instances $NUM_INSTANCES --test-split 0.5 --num-scenarios 1 10 20 30 50 75 100 --evaluated-instances $NUM_EVALUATED_INSTANCES
done
