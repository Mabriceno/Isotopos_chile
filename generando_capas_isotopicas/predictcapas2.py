import pandas as pd
import numpy as np 
from bio import Param_bioclimaticos as Pb
from netCDF4 import Dataset 
import pickle

anio = ' 2028'
#######################Importar Biocapas
filehandler = open('parametros_bioclimaticos.lec', 'rb') 
pb = pickle.load(filehandler)
grid_lat = pb.grid_lat
grid_lon = pb.grid_lon


for i in range(0,len(grid_lat)):
    if grid_lat[i] == -36.04999923706055:
        print(i)

lon=[]
for i in grid_lat:
    lon.append(grid_lon)

lon=np.array(lon)

#######################Aniadir Modelo

'''

## N Suelo N ~ + bio2 + bio14 + bio3 
Intercept = -4.598919
bio2 = -0.717209
bio14 = -0.150754
bio3 = 0.307396

capa = pb.bio2*bio2 + pb.bio14*bio14 + pb.bio3*bio3 + Intercept
title=' ($ \delta ^{15} N$) on Soil Isoscape'+anio


## C Suelo C ~ + bio4 + bio5 + bio8

Intercept = -23.345476
bio4 = 0.012882
bio5 = -0.626496
bio8 = 0.910948

capa = pb.bio4*bio4 + pb.bio5*bio5 + pb.bio8*bio8 + Intercept
title=' ($ \delta ^{13} C$) on Soil Isoscape'+anio


## N Plantas N ~ + bio8 + bio13 + bio12

Intercept = 2.960691
bio8 = 0.658448
bio13 = -0.490125
bio12 = 0.106791

capa = pb.bio8*bio8 + pb.bio13*bio13 + pb.bio12*bio12 + Intercept
title=' ($ \delta ^{15} N$) on Plant Isoscape'+anio

'''
## C Plantas C ~ + bio18 + Longitude + bio1
Intercept = 210.930572
bio18 = 0.123049
Longitude = 3.553860
bio1 = 0.985577

capa = pb.bio18*bio18 + lon*Longitude + pb.bio1*bio1 + Intercept
title=' ($ \delta ^{13} C$) on Plant Isoscape'+anio




################cortar capa
######## Generate a grid:
dataset = Dataset("tmin.nc", mode="r")
grid_lat1 = dataset.variables['lat'][:]
grid_lon1 = dataset.variables['lon'][:]
tmin = dataset.variables['tmin'][:]
tmin = tmin[6]
dataset.close()
def x(i,j,grid_lat,grid_lon,grid_lat1,grid_lon1,tmin,nan):
    lat = grid_lat[i]
    lon = grid_lon[j]
    closer_lat = min(grid_lat1, key=lambda x:abs(x-lat))
    closer_lon = min(grid_lon1, key=lambda x:abs(x-lon))
    coo_lat = list(grid_lat1).index(closer_lat)
    coo_lon = list(grid_lon1).index(closer_lon)
    #print(tmin[coo_lat][coo_lon]==tmin[0][0])
    if type(tmin[coo_lat][coo_lon])==type(tmin[0][0]):
        #print('l')
        return True
    else:
        return False
nan = tmin[30][0]
for i in range(0,len(capa)):
    for j in range(0,len(capa[i])):
        if x(i,j,grid_lat,grid_lon,grid_lat1,grid_lon1,tmin,nan):
            capa[i][j]=nan

###################################################################

###PROMEDIO
promedio = np.nanmean(capa[255:][:])
std = np.nanstd(capa[255:][:])
print(promedio)
print(std)

######################Crear capa
capa2028 = np.genfromtxt('dNsoil2028.csv', delimiter=",", skip_header=1)
grid_lat2028 = np.genfromtxt('grid_lat2028.csv', delimiter=";", skip_header=1)
grid_lon2028 = np.genfromtxt('grid_lon2028.csv', delimiter=";", skip_header=1)

import cap
#grid_lon = np.append(grid_lon,-4.695999923706054688e+01)

#np.savetxt("dNsoil2028.csv", capa[255:][:], delimiter=",")
#np.savetxt("grid_lon2028.csv", grid_lon[:], delimiter=",")
#np.savetxt("grid_lat2028.csv", grid_lat[255:], delimiter=",")
print(np.shape(grid_lat[255:]))
print(np.shape(grid_lon))
print(np.shape(capa2028))
cap.generate(grid_lat[255:],grid_lon[:],capa[255:][:],title,'Dark2')
#cap.generate(grid_lat[255:],grid_lon[:],capa2028[:],title,'Dark2')

