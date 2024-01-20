# FairDiversity

## Downloading datasets

There are two ways to get the datasets working:
### Google Drive

Download the ziped file at: https://drive.google.com/file/d/1DRjdWPWvGkUfGDajSLywRZzYRqmY-cZ_/view?usp=share_link

Extract and copy its contents into the `./datasets` for all the imports in the code to work.

### Script

Run `./datasets/init_datasets.sh` to download and initialize all datasets.
NOTE: This does not work for the YELP dataset

## Validating Results from Paper Submission
To reproduce our results, run `./run_all.sh` which will run all the experiments mentioned in `./publish/setup/`.

To validate our graphs, the readings can be found in `./publish/results`. To replot graphs run `./plot_all.sh` within `./publish`.

### Install and run the virtual env.
Note: License for gurobi is required to run a few algorithms
```
pipenv install
pipenv shell
```

### Run experiments

Experiement setup files can be found under `./experiments`, an experiment can be run using `runner_k.py` or `streamrunner_k.py`. For example:
```
python runner_k.py ./experiments/equal_kis.json
```
The result files would also be generated under ./experiments with a timestamp

### Plot experiments
```
python plot_k.py ./experiments/result_equal_kis/result_23_10_12_03_10_24.json
```
