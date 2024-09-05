#!/bin/bash

# Data for the figures from the paper are generated using the following
# Fig 3, Fig 4   python3 runner_k_no_gurobi.py ./publish/setup/micro_exp_equal.json
# Fig 5, Fig 6, Fig 9   python3 runner_k_no_gurobi.py ./publish/setup/macro_exp_equal.json
# Fig 7, Fig 8   python3 runner_k_no_gurobi.py ./publish/setup/macro_exp_prop.json
# Fig 10  python3 streamrunner_k.py ./publish/setup/macro_stream_beer_review.json

# Remove any previous runs
rm -r -f ./publish/setup/result_*

if [[ "$*" == *"bash"* ]]
then
	bash
	exit 0
fi
if [[ "$*" == *"--no-gurobi"* ]]; then
    python3 runner_k_no_gurobi.py ./publish/setup/micro_exp_equal.json
    python3 runner_k_no_gurobi.py ./publish/setup/macro_exp_equal.json
    python3 runner_k_no_gurobi.py ./publish/setup/macro_exp_prop.json
    python3 streamrunner_k.py ./publish/setup/macro_stream_beer_review.json
elif [[ "$*" == *"--only-mfd-rest-cached"* ]]; then
    python3 runner_k_only_mfd.py ./publish/setup/micro_exp_equal.json
    python3 runner_k_only_mfd.py ./publish/setup/macro_exp_equal.json
    python3 runner_k_only_mfd.py ./publish/setup/macro_exp_prop.json
    python3 streamrunner_k_only_mfd.py ./publish/setup/macro_stream_beer_review.json
else
	# make sure we have an environment file (and hopefully a license file)
	if ! [ -f /opt/IO/env ]
	then
		echo 'ERROR: missing environment file at "IO/env"!'
		echo 'Example environment file copied in to IO/example_env'
		cp /opt/gurobi1003/example_env /opt/IO/example_env 
		echo "Make sure to provide a proper license file in IO (default location in example is at IO/gurobi.lic)!"
		exit 1
	fi

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
python3 publish/plot.py ./publish/reproduced_results/micro_exp_equal.json 1.2 300
python3 publish/plot.py ./publish/reproduced_results/macro_exp_equal.json 0.2 300
python3 publish/plot.py ./publish/reproduced_results/macro_exp_prop.json 0.2 300
python3 publish/streamplot.py ./publish/reproduced_results/macro_stream_beer_review.json 300 -nolegend

# send the entire result folder back to the main machine
# cp -r publish/ /opt/IO/publish/


rm -r -f publish/plots/micro_bench/equal/
mkdir -p publish/plots/micro_bench/equal/
# After running this script the figures are related to the paper as follows:
# Fig 3 
#   Original:
#        publish/results/micro_exp_equal/diversity_vs_k.png
#   Reproduced
#       publish/reproduced_results/micro_exp_equal/diversity_vs_k.png
cp publish/reproduced_results/micro_exp_equal/diversity_vs_k.png publish/plots/micro_bench/equal/

# Fig 4
#   Original:
#       publish/results/micro_exp_equal/runtime_vs_k.png
#   Reproduced
#       publish/reproduced_results/micro_exp_equal/runtime_vs_k.png
cp publish/results/micro_exp_equal/runtime_vs_k.png publish/plots/micro_bench/equal/


rm -r -f publish/plots/macro_bench/equal/
mkdir -p publish/plots/macro_bench/equal/
# Fig 5
#   Original:
#       publish/results/macro_exp_equal/diversity_vs_k.png
#   Reproduced
#       publish/reproduced_results/macro_exp_equal/diversity_vs_k.png
cp publish/reproduced_results/macro_exp_equal/diversity_vs_k.png publish/plots/macro_bench/equal/

# Fig 6
#   Original:
#       publish/results/macro_exp_equal/runtime_vs_k.png
#   Reproduced
#       publish/reproduced_results/macro_exp_equal/runtime_vs_k.png  
cp publish/reproduced_results/macro_exp_equal/runtime_vs_k.png  publish/plots/macro_bench/equal/


rm -r -f publish/plots/macro_bench/proportional/
mkdir -p publish/plots/macro_bench/proportional/
# Fig 7
#   Original:
#       publish/results/macro_exp_prop/diversity_vs_k.png
#   Reproduced
#       publish/reproduced_results/macro_exp_prop/diversity_vs_k.png
cp publish/reproduced_results/macro_exp_prop/diversity_vs_k.png publish/plots/macro_bench/proportional/

# Fig 8
#   Original:
#       publish/results/macro_exp_prop/runtime_vs_k.png
#   Reproduced
#       publish/reproduced_results/macro_exp_prop/runtime_vs_k.png
cp publish/reproduced_results/macro_exp_prop/runtime_vs_k.png publish/plots/macro_bench/proportional/


rm -r -f publish/plots/skyline
mkdir -p publish/plots/skyline
# Fig 9
#   Original:
#       publish/results/macro_exp_equal/diversity_vs_runtime_100.png
#   Reproduced
#       publish/reproduced_results/macro_exp_equal/diversity_vs_runtime_100.png
cp publish/reproduced_results/macro_exp_equal/diversity_vs_runtime_100.png ./publish/plots/skyline

rm -r -f publish/plots/stream_beer
mkdir -p publish/plots/stream_beer
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
cp publish/results/macro_stream_beer_review/legend.png publish/plots/stream_beer
cp publish/reproduced_results/macro_stream_beer_review/streamtime_vs_k.png publish/plots/stream_beer
cp publish/reproduced_results/macro_stream_beer_review/posttime_vs_k.png publish/plots/stream_beer
cp publish/reproduced_results/macro_stream_beer_review/diversity_vs_k.png publish/plots/stream_beer

# copy the plots to the paper
# cp -r publish/plots /opt/paper

# rm -r /opt/paper


# build the paper
# cd /opt/paper && ./tectonic main.tex && cp main.pdf /opt/IO/paper.pdf

