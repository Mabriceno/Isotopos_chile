import os
from netCDF4 import Dataset 
import numpy as np
from net2 import Net

var = 'pr'
dir = '/Users/lequi/Documents/Python/generando_capas_bioclimaticas/rcp85/'+var+'/CR2-RegCM4-6/MPI-M-MPI-ESM-MR/r1i1p1/'
contenido = os.listdir(dir)
nc_files= []
for fichero in contenido:
    if os.path.isfile(os.path.join(dir, fichero)) and fichero.endswith('.nc'):
        nc_files.append(fichero)
nc_files.sort()

cube=[]
print(cube)

dataset = Dataset(dir+nc_files[13], mode="r")
print(dataset.variables[var])
print(dataset.variables['time'])
print(dataset.variables['lat'])
print(dataset.variables['lon'])
dataset.close()

for nc_name in nc_files:
    dataset = Dataset(dir+nc_name, mode="r")
    z = dataset.variables[var][:]
    time = dataset.variables['time'][:]
    lat = dataset.variables['lat'][:]
    lon = dataset.variables['lon'][:]
    dataset.close()
    h = (24*60*60*30)*z
    #h = z + np.full(np.shape(z),-273.15)
    for i in range(0,len(h)):
        cube.append(h[i])



file = Net(var, cube, lon, lat, len(nc_files))
file.setup()