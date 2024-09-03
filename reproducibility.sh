#!/bin/bash

# Remove any previous runs
rm -r -f ./publish/setup/result_*

# Check if the --no-gurobi flag is present
if [[ "$@" == *"--no-gurobi"* ]]; then
    python3 runner_k_no_gurobi.py ./publish/setup/macro_exp_equal.json
    python3 runner_k_no_gurobi.py ./publish/setup/macro_exp_prop.json
    python3 runner_k_no_gurobi.py ./publish/setup/micro_exp_prop.json
    python3 runner_k_no_gurobi.py ./publish/setup/macro_exp_equal.json
    python3 streamrunner_k.py ./publish/setup/macro_stream_beer_review.json
else
    python3 runner_k.py ./publish/setup/macro_exp_equal.json
    python3 runner_k.py ./publish/setup/macro_exp_prop.json
    python3 runner_k.py ./publish/setup/micro_exp_prop.json
    python3 runner_k.py ./publish/setup/macro_exp_equal.json
    python3 streamrunner_k.py ./publish/setup/macro_stream_beer_review.json
fi


# Create reproduced results directtory
rm -r -f ./publish/reproduced_results
mkdir ./publish/reproduced_results

# Copy reproduced result files
cp ./publish/setup/result_macro_exp_equal/*.json ./publish/reproduced_results/macro_exp_equal.json
cp ./publish/setup/result_macro_exp_prop/*.json ./publish/reproduced_results/macro_exp_prop.json
cp ./publish/setup/result_micro_exp_equal/*.json ./publish/reproduced_results/micro_exp_equal.json
cp ./publish/setup/result_micro_exp_prop/*.json ./publish/reproduced_results/micro_exp_prop.json
cp ./publish/setup/result_macro_stream_beer_review/*.json ./publish/reproduced_results/macro_stream_beer_review.json
cp ./publish/setup/result_macro_stream_popsim/*.json ./publish/reproduced_results/macro_stream_popsim.json

# Plot the reproduced results
python3 ./publish/plot.py ./publish/reproduced_results/macro_exp_equal.json 0.2 300
python3 ./publishplot.py ./publish/reproduced_results/macro_exp_prop.json 0.2 300
python3 ./publishplot.py ./publish/reproduced_results/micro_exp_equal.json 1.2 300
python3 ./publishplot.py ./publish/reproduced_results/micro_exp_prop.json1.2 300S
python3 ./publishstreamplot.py ./publish/reproduced_results/macro_stream_beer_review.json 300 -nolegend
python3 ./publishstreamplot.py ./publish/reproduced_results/macro_stream_popsim.json 300 -nolegend