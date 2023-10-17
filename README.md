# FairDiversity

## Downloading datasets

### Google Drive

Download the ziped file at: https://drive.google.com/file/d/1d79JBIssbDwIMWV3v_h154ouITwGp4eu/view?usp=share_link

Extract and opy its contents into the `./datasets` for all the imports in the code to work.

### Script

Run `./datasets/init_datasets.sh` to download and initialize all datasets.

## Running algorithm tests


### Install and run the virtual env.
Note: License for gurobi is required to run a few algorithms
```
pipenv install
pipenv shell
```

### Run experiments

Experiement setup files can be found under `./experiments`, an experiment can be run using `runner_k.py`. For example:
```
python runner_k.py ./experiments/equal_kis.json
```
The result files would also be generated under ./exepriments with a timestamp

### Plot experiments
```
python plot_k.py ./experiments/result_equal_kis/result_23_10_12_03_10_24.json
```
