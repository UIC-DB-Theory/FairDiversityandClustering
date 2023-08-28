import json

dictionary = {

    "filename" : "adult.data",
    
    # Note that these need to be listed in order to what they are in the dataset file
    "allFields" : [

        "age",
        "workclass",
        "fnlwgt",  # what on earth is this one?
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
        "yearly-income",
    ],

    "colorFields" : [
        'race', 
        'sex'
    ],

    "feature_fields" : [
        'age', 
        'capital-gain', 
        'capital-loss', 
        'hours-per-week', 
        'fnlwgt', 
        'education-num'
    ],
}

# Serializing json
json_object = json.dumps(dictionary, indent=8)
 
# Writing to sample.json
with open("dataset.json", "w") as outfile:
    outfile.write(json_object)