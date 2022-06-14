import numpy as np

from netCDF4 import Dataset 

from net import Net
from calendar import monthrange
'''#dataset = Dataset("data/CR2MET_v2.0_mon_1979_2019_005deg/tmin.nc", mode="r")
dataset = Dataset("bio1.nc", mode="r")

grid_lat = dataset.variables['lat'][:]
grid_lon = dataset.variables['lon'][:]
z = dataset.variables['bio1'][0:2]

print(dataset)
file = Net('bio1', z, grid_lon, grid_lat, 2)
file.setup()'''
'''
num_days = monthrange(2012, 2)[1]
print(num_days)'''

def call_var(name):
    dataset = Dataset('/Users/bmati/Desktop/proyecto296/generando_capas_bioclimaticas/data/cr2met_v1_mon_1979_2016/pr.nc', mode="r")
    z = dataset.variables[name][:]
    print(dataset.variables[name])
    print(dataset.variables['time'])
    print(dataset.variables['lat'])
    print(dataset.variables['lon'])


call_var('pr')