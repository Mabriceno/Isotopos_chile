import pandas as pd
import numpy as np 
from bioclimatic2 import Param_bioclimaticos as Pb
from netCDF4 import Dataset 
from net import Net
import datetime

######### call nc
def call_var(name):
    dataset = Dataset('data/rcp26AllInOne/'+name+'.nc', mode="r")
    z = dataset.variables[name][:]
    print(dataset.variables[name])
    print(dataset.variables['time'])
    print(dataset.variables['lat'])
    print(dataset.variables['lon'])

    dataset.close()
    return z

######## Generate a grid:
dataset = Dataset("data/cr2met_v1_mon_1979_2016/tmin.nc", mode="r")
grid_lat = dataset.variables['lat'][:]
grid_lon = dataset.variables['lon'][:]
dataset.close()

names = ['tas','tasmin','tasmax','pr']
variables = []

'''def nan(var):
    for i in range(0,len(var)):
        for j in range(0,len(var[i])):
            for k in range(0,len(var[i][j])):
                if var[i][j][k]==-9999.0:
                    
                    var[i][j][k]=np.nan
    return var'''

n_years = 44

for name in names:
    variable = call_var(name)
    #variable = nan(variable)
    variable = variable.astype(float)
    variable[variable==-9999.0] = np.nan
    variable = np.split(variable,n_years)
    variables.append(variable)

anio = 2007
names = ['bio1','bio2','bio3','bio4','bio5','bio6','bio7','bio8','bio9','bio10','bio11','bio12','bio13','bio14','bio15','bio16','bio17','bio18','bio19']

cubes = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
pb_years = []
for year in range(0,n_years):
    print(datetime.datetime.now())
    print('Comienzo calculo de variable bioclimatica anio '+str(anio))
    pb = Pb(variables[0][year], variables[1][year], variables[2][year], variables[3][year],grid_lat,grid_lon, np.shape(variables[0][0][0]))
    pb.setup()
    bios = pb.get_bio() #bios contiene las 19 bioclimaticas del anio n  
    
    for i in range(0, len(cubes)):
        cubes[i].append(bios[i])
        print(names[i]+'anio '+str(anio)+' listo !')
    anio+=1

for i in range(0, len(cubes)):
    file = Net(names[i], cubes[i], grid_lon, grid_lat, n_years)
    file.setup()









        



