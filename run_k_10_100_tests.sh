#!/bin/bash

python3 exp_t_div_vs_k_runner_v3.py ./experiments/all_popsim_k_10_100.json
python3 exp_t_div_vs_k_runner_v3.py ./experiments/all_adult_k_10_100.json
python3 exp_t_div_vs_k_runner_v3.py ./experiments/all_diabetes_k_10_100.json
python3 exp_t_div_vs_k_runner_v3.py ./experiments/all_census_k_10_100.json
python3 exp_t_div_vs_k_runner_v3.py ./experiments/all_census_k_10_100_25d.json

