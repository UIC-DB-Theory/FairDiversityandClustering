import json
import csv

metadata_file = "yelp_business.metadata"
metadata = {
    "filename" : "yelp_business.data",
    "url" : "https://www.yelp.com/dataset/documentation/main"
}


# fields can be found in beeradvocate.txt
fields = [
                "beer/name",
                "x",
                "y",
                "z"
        ]
data = []

# ---------------Read original dataset file--------------
with open('Beeradvocate.txt', 'r') as file:
    lines = file.readlines()
    invalid = 0
    for i in range(0, len(lines)):
        try:
            if "beer/name" in lines[i]:
                # The next 12 lines are part of the same review
                name = lines[i].split(':')[1].strip()
                beerId = int(lines[i+1].split(':')[1].strip())
                brewerId = int(lines[i+2].split(':')[1].strip())
                abv = float(lines[i+3].split(':')[1].strip())
                style = lines[i+4].split(':')[1].strip()
                appearance = float(lines[i+5].split(':')[1].strip())
                aroma = float(lines[i+6].split(':')[1].strip())
                palate = float(lines[i+7].split(':')[1].strip())
                taste = float(lines[i+8].split(':')[1].strip())
                overall = float(lines[i+9].split(':')[1].strip())
                time = int(lines[i+10].split(':')[1].strip())
                profileName = lines[i+11].split(':')[1].strip()
                text = lines[i+12].split(':')[1].strip()
                
                if 'Ale' in name:
                    category = 'Ale'
                elif 'Lager' in name:
                    category = 'Lager'
                else:
                    category = 'Other'

                data.append(
                    [name,
                    beerId,
                    brewerId,
                    abv,
                    style,
                    appearance,
                    aroma,
                    palate,
                    taste,
                    overall,
                    time,
                    profileName,
                    text,
                    category]
                )
            
        except:
            invalid += 1
            continue
    print('Invalid points: ', invalid)
    print('Valid points read: ', len(data))
        

# ---------------Create file beer_reviews.metadata--------------
metadata["fields"] = fields

# Write to metadata file
json_object = json.dumps(metadata, indent=4)
with open(metadata_file, "w") as outfile:
    outfile.write(json_object)


# ---------------Create file beer_reviews.metadata---------------
with open(metadata["filename"], "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(data)









import json

with open("test.json") as json_file:
    json_data = json.load(json_file)
    print(json_data)


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