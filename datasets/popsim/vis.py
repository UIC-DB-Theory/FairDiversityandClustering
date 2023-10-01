import json
import csv

data = []
# ---------------Read original dataset file--------------
# P1_003N, P1_004N, P1_005N, P1_006N, P1_007N
# white, black or African American, American Indian and Alaska Native, Asian, Native Hawaiian, Other Pacific Islander
race_mappings = {
    'P1_003N' : 'White',
    'P1_004N' : 'Black/African American',
    'P1_005N' : 'African American/Alaska Native',
    'P1_006N' : 'Asian',
    'P1_007N' : 'Native Hawaiian/Other Pacific Islander',
}
with open("./popsim_5m.csv", 'r') as file:
    csvreader = csv.reader(file)
    i = -1
    for row in csvreader:
        i = i + 1
        if i == 0:
            fields = ['race', 'lat', 'lon']
            data.append(fields)
            continue
        # only append rows with race mappings
        if row[2] in race_mappings:
            data.append([race_mappings[row[2]], float(row[3]), float(row[4])])


# ---------------Create file diabetes.metadata--------------
# ---------------Create file diabetes.data---------------
with open('popsimvis.data', "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(data)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("popsimvis.data")

print(df.head())

BBox = (df.lon.min(), df.lon.max(), df.lat.min(), df.lat.max())

print(BBox)
race_color = {
    'White' : 'red',
    'Black/African American' : 'blue',
    'African American/Alaska Native' : 'yellow',
    'Asian' : 'blue',
    'Native Hawaiian/Other Pacific Islander' : 'green',
}

import pandas as pd
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(12,9))
m = Basemap(projection = 'mill', llcrnrlat = BBox[0], urcrnrlat = BBox[1], llcrnrlon = BBox[2], urcrnrlon = BBox[3], resolution = 'c')
m.drawcoastlines()
m.drawcountries(color='gray')
m.drawstates(color='gray')
for race in race_color:
    data_for_race = df.loc[df['race'] == race]
    lons = data_for_race['lon'].to_list()
    lats = data_for_race ['lat'].to_list()
    m.scatter(lats, lons, latlon = True, s = 1, c = race_color[race], marker = 'o', alpha = 1, label=race)

plt.legend(title = f'Legend', bbox_to_anchor=(1.05, 1.0), loc='upper left')
plt.savefig(f'vis', dpi=300, bbox_inches='tight')