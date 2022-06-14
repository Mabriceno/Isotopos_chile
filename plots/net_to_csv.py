from netCDF4 import Dataset
import pandas as pd
import numpy as np

ISOTOPO = 'N_Soil'
RCP = '26'
def call_var(name):
    dataset = Dataset('data/'+RCP+'/'+name+'.nc', mode="r")
    z = dataset.variables[name][:]
    grid_lat = dataset.variables['lat'][:]
    grid_lon = dataset.variables['lon'][:]
    dataset.close()

    return z, grid_lat, grid_lon

year = 2007
z, grid_lat, grid_lon = call_var(ISOTOPO)

date = []
lat = []
lon = []
value = []

for t in range(0, len(z)):
    for i in range(0,len(z[0])):
        for j in range(0, len(z[0][0])):
            if j != np.nan:
                
                date.append(year)
                lat.append(grid_lat[i])
                lon.append(grid_lon[j])
                value.append(z[t][i][j])
    year += 1

df = {'date':date,
    'lat':lat,
    'lon':lon,
    ISOTOPO: value}

df = pd.DataFrame(data=df)

print(np.shape(df))
df = df.dropna(subset=[ISOTOPO])
print(np.shape(df))


df.to_csv(ISOTOPO+RCP+'.csv',index=False)