import json
import csv
"""
beer/name: Sausa Weizen
beer/beerId: 47986
beer/brewerId: 10325
beer/ABV: 5.00
beer/style: Hefeweizen
review/appearance: 2.5
review/aroma: 2
review/palate: 1.5
review/taste: 1.5
review/overall: 1.5
review/time: 1234817823
review/profileName: stcules
review/text: A lot of foam. But a lot.	In the smell some banana, and then lactic and tart. Not a good start.	Quite dark orange in color, with a lively carbonation (now visible, under the foam).	Again tending to lactic sourness.	Same for the taste. With some yeast and banana.	
"""
metadata_file = "beer_reviews.metadata"
metadata = {
    "filename" : "beer_reviews.data",
    "url" : "none"
}


# fields can be found in beeradvocate.txt
fields = [
                "beer/name",
                "beer/brewerId",
                "beer/ABV",
                "beer/style",
                "review/appearance",
                "review/aroma",
                "review/palate",
                "review/taste",
                "review/overall",
                "review/time",
                "review/profileName",
                "review/text"
        ]
data = []

# ---------------Read original dataset file--------------
with open('beeradvocate.txt', 'r') as file:
    lines = file.readlines()
    counter = 0
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
                    text]
                )
        except:
            counter += 1
            continue
    print('Invalid points: ', counter)
        

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