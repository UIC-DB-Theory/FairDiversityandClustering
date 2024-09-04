#!/bin/bash

# Remove any previous runs
rm -r -f ./publish/setup/result_*


if [[ "$*" == *"--no-gurobi"* ]]; then
    python3 runner_k_no_gurobi.py ./publish/setup/micro_exp_equal.json
    python3 runner_k_no_gurobi.py ./publish/setup/macro_exp_equal.json
    python3 runner_k_no_gurobi.py ./publish/setup/macro_exp_prop.json
    python3 streamrunner_k.py ./publish/setup/macro_stream_beer_review.json
elif [[ "$*" == *"--only-mfd-rest-cached"* ]]; then
    python3 runner_k_only_mfd.py ./publish/setup/micro_exp_equal.json
    python3 runner_k_only_mfd.py ./publish/setup/macro_exp_equal.json
    python3 runner_k_only_mfd.py ./publish/setup/macro_exp_prop.json
    python3 streamrunner_k.py ./publish/setup/macro_stream_beer_review.json
else
    python3 runner_k.py ./publish/setup/micro_exp_equal.json
    python3 runner_k.py ./publish/setup/macro_exp_equal.json
    python3 runner_k.py ./publish/setup/macro_exp_prop.json
    python3 streamrunner_k.py ./publish/setup/macro_stream_beer_review.json
fi

# Create reproduced results directtory
rm -r -f ./publish/reproduced_results
mkdir ./publish/reproduced_results

# Copy reproduced result files
cp ./publish/setup/result_micro_exp_equal/*.json ./publish/reproduced_results/micro_exp_equal.json
cp ./publish/setup/result_macro_exp_equal/*.json ./publish/reproduced_results/macro_exp_equal.json
cp ./publish/setup/result_macro_exp_prop/*.json ./publish/reproduced_results/macro_exp_prop.json
cp ./publish/setup/result_macro_stream_beer_review/*.json ./publish/reproduced_results/macro_stream_beer_review.json

# Plot the reproduced results
python3 ./publish/plot.py ./publish/reproduced_results/micro_exp_equal.json 1.2 300
python3 ./publish/plot.py ./publish/reproduced_results/macro_exp_equal.json 0.2 300
python3 ./publish/plot.py ./publish/reproduced_results/macro_exp_prop.json 0.2 300
python3 ./publish/streamplot.py ./publish/reproduced_results/macro_stream_beer_review.json 300 -nolegend