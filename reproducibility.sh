#!/bin/bash

# Data for the figures from the paper are generated using the following
# Fig 3, Fig 4   python3 runner_k_no_gurobi.py ./publish/setup/micro_exp_equal.json
# Fig 5, Fig 6, Fig 9   python3 runner_k_no_gurobi.py ./publish/setup/macro_exp_equal.json
# Fig 7, Fig 8   python3 runner_k_no_gurobi.py ./publish/setup/macro_exp_prop.json
# Fig 10  python3 streamrunner_k.py ./publish/setup/macro_stream_beer_review.json

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

# Create reproduced results directory
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

# After running this script the figures are related to the paper as follows:
# Fig 3 
#   Original:
#        publish/results/micro_exp_equal/diversity_vs_k.png
#   Reproduced
#       publish/reproduced_results/micro_exp_equal/diversity_vs_k.png
# Fig 4
#   Original:
#       publish/results/micro_exp_equal/runtime_vs_k.png
# Fig 5
#   Original:
#       publish/results/macro_exp_equal/diversity_vs_k.png
#   Reproduced
#       publish/reproduced_results/macro_exp_equal/diversity_vs_k.png
# Fig 6
#   Original:
#       publish/results/macro_exp_equal/runtime_vs_k.png
#   Reproduced
#       publish/reproduced_results/macro_exp_equal/runtime_vs_k.png  
# Fig 7
#   Original:
#       publish/results/macro_exp_prop/diversity_vs_k.png
#   Reproduced
#       publish/reproduced_results/macro_exp_prop/diversity_vs_k.png
# Fig 8
#   Original:
#       publish/results/macro_exp_prop/runtime_vs_k.png
#   Reproduced
#       publish/reproduced_results/macro_exp_prop/runtime_vs_k.png
# Fig 9
#   Original:
#       publish/results/macro_exp_equal/diversity_vs_runtime_100.png
#   Reproduced
#       publish/reproduced_results/macro_exp_equal/diversity_vs_runtime_100.png
# Fig 10
#   Original:
#       publish/results/macro_stream_beer_review/legend.png
#       publish/results/macro_stream_beer_review/streamtime_vs_k.png
#       publish/results/macro_stream_beer_review/posttime_vs_k.png
#       publish/results/macro_stream_beer_review/diversity_vs_k.png
#   Reproduced
#       publish/results/macro_stream_beer_review/legend.png (use same legend)
#       publish/reproduced_results/macro_stream_beer_review/streamtime_vs_k.png
#       publish/reproduced_results/macro_stream_beer_review/posttime_vs_k.png
#       publish/reproduced_results/macro_stream_beer_review/diversity_vs_k.png