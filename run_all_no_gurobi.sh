#!/bin/bash
python3 runner_k_no_gurobi.py ./publish/setup/macro_exp_equal.json
python3 runner_k_no_gurobi.py ./publish/setup/macro_exp_prop.json
python3 runner_k_no_gurobi.py ./publish/setup/micro_exp_prop.json
python3 runner_k_no_gurobi.py ./publish/setup/macro_exp_equal.json
python3 streamrunner_k.py ./publish/setup/macro_stream_beer_review.json
