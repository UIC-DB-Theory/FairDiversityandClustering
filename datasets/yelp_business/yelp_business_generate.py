import json
import csv
import math

metadata_file = "yelp_business.metadata"
metadata = {
    "filename" : "yelp_business.data",
    "url" : "https://www.yelp.com/dataset/documentation/main"
}


fields = [
                "state_category",
                "x",
                "y",
                "z"
        ]
data = []

# ---------------Read original dataset file--------------
with open('yelp_academic_dataset_business.json', 'r') as file:
    lines = file.readlines()
    for i in range(0, len(lines)):
        business = json.loads(lines[i])
        state_abbreviation = business['state']
        state_category
        if state_abbreviation.upper() in ["MI", "OH", "PA", "IN", "IL"]:
            state_category = "Industrial/Manufacturing"
        elif state_abbreviation.upper() in ["CA", "WA", "MA", "TX"]:
            state_category = "Technology/Innovation"
        elif state_abbreviation.upper() in ["IA", "KS", "NE", "SD", "ND"]:
            state_category = "Agriculture"
        elif state_abbreviation.upper() in ["TX", "AK", "WY", "ND"]:
            state_category = "Energy/Natural Resources"
        elif state_abbreviation.upper() in ["NY", "DE", "CT", "IL"]:
            state_category = "Finance/Services"
        elif state_abbreviation.upper() in ["FL", "NV", "HI", "CA"]:
            state_category = "Tourism/Service Industries"
        else:
            state_category = "Other"
        
        lat = business['latitude']
        lon = business['longitude']
        x = 6371*math.cos(lat)*math.cos(lon)
        y = 6371*math.cos(lat)*math.sin(lon)
        z = 6371*math.sin(lat)
        data.append([state_category, x, y, z])
        
        

# ---------------Create file yelp_business.metadata--------------
metadata["fields"] = fields

# Write to metadata file
json_object = json.dumps(metadata, indent=4)
with open(metadata_file, "w") as outfile:
    outfile.write(json_object)


# ---------------Create file yelp_business.data---------------
with open(metadata["filename"], "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(data)



'''
{
    // string, 22 character unique string business id
    "business_id": "tnhfDv5Il8EaGSXZGiuQGg",

    // string, the business's name
    "name": "Garaje",

    // string, the full address of the business
    "address": "475 3rd St",

    // string, the city
    "city": "San Francisco",

    // string, 2 character state code, if applicable
    "state": "CA",

    // string, the postal code
    "postal code": "94107",

    // float, latitude
    "latitude": 37.7817529521,

    // float, longitude
    "longitude": -122.39612197,

    // float, star rating, rounded to half-stars
    "stars": 4.5,

    // integer, number of reviews
    "review_count": 1198,

    // integer, 0 or 1 for closed or open, respectively
    "is_open": 1,

    // object, business attributes to values. note: some attribute values might be objects
    "attributes": {
        "RestaurantsTakeOut": true,
        "BusinessParking": {
            "garage": false,
            "street": true,
            "validated": false,
            "lot": false,
            "valet": false
        },
    },

    // an array of strings of business categories
    "categories": [
        "Mexican",
        "Burgers",
        "Gastropubs"
    ],

    // an object of key day to value hours, hours are using a 24hr clock
    "hours": {
        "Monday": "10:00-21:00",
        "Tuesday": "10:00-21:00",
        "Friday": "10:00-21:00",
        "Wednesday": "10:00-21:00",
        "Thursday": "10:00-21:00",
        "Sunday": "11:00-18:00",
        "Saturday": "10:00-21:00"
    }
}
'''