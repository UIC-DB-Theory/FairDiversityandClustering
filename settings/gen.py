import json

dictionary = {
    
    "setup" : {

        # k goes from 25 to 351 at increments of 50
        "k" : [25, 351, 50],

        # Size of the coreset is this times k
        "coreset_size_factor" : 10,

        # number of runs for each algorithm
        "num_runs": 1
    },
    
    # Algorithms to run with the above setup
    "algorithms" : {

        # use_precomputed_coreset
        # 1 -> Yes
        # 0 -> No
        # 2 -> Both


        "FFMD_LP" : {
            "use_precomputed_coreset" : 1
        },
        "FMMD_MWU" : {
            "use_precomputed_coreset" : 1
        },
        "FairFlow": {
            "use_precomputed_coreset" : 0
        },
        "FairGreedyFlow":{
            "use_precomputed_coreset" : 0
        },
        "FMMD-S":{
            "use_precomputed_coreset" : 0
        },
        "SFDM-2":{
            "use_precomputed_coreset" : 0
        }
    },
}

# Serializing json
json_object = json.dumps(dictionary, indent=8)
 
# Writing to sample.json
with open("dataset.json", "w") as outfile:
    outfile.write(json_object)