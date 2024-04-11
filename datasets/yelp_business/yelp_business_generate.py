import json
import csv
import math

metadata_file = "yelp_business.metadata"
metadata = {
    "filename" : "yelp_business.data",
    "url" : "https://www.yelp.com/dataset/documentation/main"
}


fields = [
                "rating_category",
                "x",
                "y",
                "z"
        ]
data = []

# ---------------Read original dataset file--------------

points_per_color = {}
with open('yelp_academic_dataset_business.json', 'r') as file:
    lines = file.readlines()
    for i in range(0, len(lines)):
        business = json.loads(lines[i])
        rating = float(business['stars'])

        if rating >= 4.5:
            rating_category = "[4.5,max]"
        elif rating >= 3.5:
            rating_category = "[3.5,4.5)"
        else:
            rating_category = "[min,3.5)"

        if rating_category not in points_per_color:
            points_per_color[rating_category] = 1
        else:
            points_per_color[rating_category] += 1                                
        
        lat = business['latitude']
        lon = business['longitude']
        x = 6371*math.cos(lat)*math.cos(lon)
        y = 6371*math.cos(lat)*math.sin(lon)
        z = 6371*math.sin(lat)
        data.append([rating_category, x, y, z])

metadata['points_per_color'] = points_per_color
        

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