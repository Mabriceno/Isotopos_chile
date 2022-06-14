import pandas as pd
import numpy as np 
from bioclimatic2 import Param_bioclimaticos as Pb
from netCDF4 import Dataset 
from net import Net
import datetime
import os 
from tqdm import tqdm
from calendar import monthrange


names = ['tas','tasmin','tasmax','pr']

#### Crea un diccionario con los archivos nc
l = []
for var in names:
    dir = '/Users/bmati/Desktop/proyecto296/generando_capas_bioclimaticas/data/rcp26/'+var+'/CR2-RegCM4-6/MPI-M-MPI-ESM-MR/r1i1p1/'
    contenido = os.listdir(dir)
    nc_files= []
    for fichero in contenido:
        if os.path.isfile(os.path.join(dir, fichero)) and fichero.endswith('.nc'):
            nc_files.append(fichero)
    nc_files.sort()
    l.append(nc_files)
names_files = {'tas':l[0],'tasmin':l[1],'tasmax':l[2],'pr':l[3],}

######### call nc
def call_var(name,i):
    dataset = Dataset('/Users/bmati/Desktop/proyecto296/generando_capas_bioclimaticas/data/rcp26/'+name+'/CR2-RegCM4-6/MPI-M-MPI-ESM-MR/r1i1p1/'+names_files[name][i], mode="r")
    z = dataset.variables[name][:]
    #print(dataset.variables[name])
    #print(dataset.variables['time'])
    #print(dataset.variables['lat'])
    #print(dataset.variables['lon'])

    dataset.close()
    return z

def arreglo_pr(z, year):
    z_=[]
    for i in range(0,len(z)):
        num_days = monthrange(2012, i+1)[1]
        m = (24*60*60*num_days)*z[i]
        z_.append(m)
    return np.array(z_)

def arreglo_t(z):
    return z + np.full(np.shape(z),-273.15)


######## Generate a grid:
dataset = Dataset('/Users/bmati/Desktop/proyecto296/generando_capas_bioclimaticas/data/rcp26/pr/CR2-RegCM4-6/MPI-M-MPI-ESM-MR/r1i1p1/'+names_files['pr'][0], mode="r")
grid_lat = dataset.variables['lat'][:]
grid_lon = dataset.variables['lon'][:]
dataset.close()

n_years = 44

anio = 2007
names_bio = ['bio1','bio2','bio3','bio4','bio5','bio6','bio7','bio8','bio9','bio10','bio11','bio12','bio13','bio14','bio15','bio16','bio17','bio18','bio19']

cubes = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
pb_years = []

for i in tqdm(range(0,n_years)):
    variables = []
    for name in names:
        variable = call_var(name,i)
        #print(np.shape(variable))
        #variable = nan(variable)
        variable = variable.astype(float)
        variable[variable==1e+20] = np.nan
        if name == 'pr':
            variable = arreglo_pr(variable, anio+i)

        else:
            variable = arreglo_t(variable)

        variables.append(variable)
    #print(np.shape(variables))
    #print(datetime.datetime.now())
    print('Comienzo calculo de variable bioclimatica anio '+str(anio))
    pb = Pb(variables[0], variables[1], variables[2], variables[3],grid_lat,grid_lon, np.shape(variables[0][0]))
    pb.setup()
    bios = pb.get_bio() #bios contiene las 19 bioclimaticas del anio n  
    
    for i in range(0, len(cubes)):
        cubes[i].append(bios[i])
        #print(names_bio[i]+'anio '+str(anio)+' listo !')
    anio+=1

for i in tqdm(range(0, len(cubes))):
    file = Net(names_bio[i], cubes[i], grid_lon, grid_lat, n_years)
    file.setup()









        



