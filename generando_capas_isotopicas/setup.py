import pandas as pd
import numpy as np 
from netCDF4 import Dataset 
import pickle
from net import Net


n_times = 44

def call_var(name):
    dataset = Dataset('/Users/bmati/Desktop/proyecto296/generando_capas_isotopicas/Data/bioclimatic_Reg26_2007_2050/'+name+'.nc', mode="r")
    z = dataset.variables[name][:]
    #print(dataset.variables[name])
    #print(dataset.variables['time'])
    grid_lat = dataset.variables['lat'][:]
    grid_lon = dataset.variables['lon'][:]

    dataset.close()
    return z, grid_lat, grid_lon


def NSoil(diccionario, i):
    Intercept = -4.598919
    bio2 = -0.717209
    bio14 = -0.150754
    bio3 = 0.307396

    z =  diccionario['bio2'][i]*bio2 + diccionario['bio14'][i]*bio14 + diccionario['bio3'][i]*bio3 + Intercept
    return z
   

def NPlant(diccionario, i):
    Intercept = 2.960691
    bio8 = 0.658448
    bio13 = -0.490125
    bio12 = 0.106791

    return diccionario['bio8'][i]*bio8 + diccionario['bio13'][i]*bio13 + diccionario['bio12'][i]*bio12 + Intercept
  

def CSoil(diccionario, i):
    Intercept = -23.345476
    bio4 = 0.012882
    bio5 = -0.626496
    bio8 = 0.910948

    return diccionario['bio4'][i]*bio4 + diccionario['bio5'][i]*bio5 + diccionario['bio8'][i]*bio8 + Intercept
    
def CPlant(diccionario, i, lon):
    Intercept = 210.930572
    bio18 = 0.123049
    Longitude = 3.553860
    bio1 = 0.985577

    return diccionario['bio18'][i]*bio18 + lon*Longitude + diccionario['bio1'][i]*bio1 + Intercept


#list_bio= ['bio1','bio2','bio3','bio4','bio5','bio8','bio12','bio13','bio14','bio18']
list_bio= ['bio2','bio3','bio14']

#importar capas bioclimaticas
'''Crea un diccionario de las variables bioclimaticas conde cada etiqueta represanta un Bio_i y contiene un tensor de capaxtiempo'''

#llamar

print('cortando capas')
AllCapas = {}
for i in list_bio:
    AllCapas[i] = []

for name in list_bio:
    capa, grid_lat, grid_lon = call_var(name)


    lons=[]
    for i in grid_lat[255:455]:
        lons.append(grid_lon[260:344])
    lons=np.array(lons)
    

    #####Cortar
    dataset = Dataset("tmin.nc", mode="r")
    grid_lat1 = dataset.variables['lat'][:]
    grid_lon1 = dataset.variables['lon'][:]
    tmin = dataset.variables['tmin'][:]
    tmin = tmin[6]
    dataset.close()
    def x(i,j,grid_lat,grid_lon,grid_lat1,grid_lon1,tmin):
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
    #nan = tmin[30][0]
    cubo = []
    print(np.shape(capa))
    for n in range(0,n_times):
        cubo_i = []
        cont = 0
        for i in range(255,455):#len(capa[n]
            cubo_j = []
            
            for j in range(260,344):#len(capa[n][i])
                if x(i,j,grid_lat,grid_lon,grid_lat1,grid_lon1,tmin):
                    #capa[n][i][j]=np.nan
                    cubo_j.append(np.nan)
                    cont += 1
                else:
                    cubo_j.append(capa[n][i][j])
            cubo_i.append(cubo_j)
        cubo.append(cubo_i)
        print(cont)
        print(2007+n)
    #####

    #[255:455][260:344])

    AllCapas[name] = np.array(cubo)

grid_lat = grid_lat[255:455]
grid_lon = grid_lon[260:344]

for i in list_bio:
    print(np.shape(AllCapas[i]))

print('generando isocpas')
for i in range(0, n_times):
    NS, NP, CS, CP = [],[],[],[]
    NS.append(NSoil(AllCapas, i))
    #NP.append(NPlant(AllCapas, i))
    #CS.append(CSoil(AllCapas, i))
    #CP.append(CPlant(AllCapas, i, lons))

list_name = ['N_Soil', 'N_Plant', 'C_Soil', 'C_Plant']
list_isocapas = [NS,NP,CS,CP]

#CREA NET
'''print('generando nc')
for i in range(1,4):###OJO
    file = Net(list_name[i], list_isocapas[i], grid_lon, grid_lat, n_times)
    file.setup()'''

file = Net(list_name[0], list_isocapas[0], grid_lon, grid_lat, n_times)
file.setup()

