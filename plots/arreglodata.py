import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np
from netCDF4 import Dataset
import histo
import series




def call_var(name,rcp):
    dataset = Dataset('data/'+rcp+'/'+name+'.nc', mode="r")
    z = dataset.variables[name][:]
    print(np.shape(z))
    #print(dataset.variables[name])
    #print(dataset.variables['time'])
    grid_lat = dataset.variables['lat'][:]
    grid_lon = dataset.variables['lon'][:]

    dataset.close()
    return z, grid_lat, grid_lon


rcp = '85'
list_name = ['C_Plant','C_Soil','N_Plant','N_Soil']

z , lat, lon  = call_var('C_Plant', rcp)

print(np.shape(z))
print(np.shape(lat))
print(lat)


##sur: <-29 lat[0:78]
##norte: >-25 lat[122:len(lat)]

for i in range(0, len(lat)):
    if lat[i]>-25.15 and lat[i]<-24.99:
        print(i)