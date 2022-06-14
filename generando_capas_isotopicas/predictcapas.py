import pandas as pd
import numpy as np 
from bio import Param_bioclimaticos as Pb
from netCDF4 import Dataset 
import pickle
import matplotlib.pyplot as plt

anio = '2018'
#######################Importar Biocapas
filehandler = open('parametros_bioclimaticos.lec', 'rb') 
pb = pickle.load(filehandler)
grid_lat = pb.grid_lat
grid_lon = pb.grid_lon

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
title=' ($ \delta ^{15} N$) on Soil Isoscape '+anio

'''
## C Suelo C ~ + bio4 + bio5 + bio8

Intercept = -23.345476
bio4 = 0.012882
bio5 = -0.626496
bio8 = 0.910948

capa = pb.bio4*bio4 + pb.bio5*bio5 + pb.bio8*bio8 + Intercept
title=' ($ \delta ^{13} C$) on Soil Isoscape'+anio

'''
## N Plantas N ~ + bio8 + bio13 + bio12

Intercept = 2.960691
bio8 = 0.658448
bio13 = -0.490125
bio12 = 0.106791

capa = pb.bio8*bio8 + pb.bio13*bio13 + pb.bio12*bio12 + Intercept
title=' ($ \delta ^{15} N$) on Plant Isoscape'+anio


## C Plantas C ~ + bio18 + Longitude + bio1
Intercept = 210.930572
bio18 = 0.123049
Longitude = 3.553860
bio1 = 0.985577

capa = pb.bio18*bio18 + lon*Longitude + pb.bio1*bio1 + Intercept
title=' ($ \delta ^{13} C$) on Plant Isoscape'+anio
'''

###PROMEDIO
promedio = np.nanmean(capa[419:][:])
std = np.nanstd(capa[419:][:])
print(promedio)
print(std)

print(np.shape(capa))
######################Crear capa
import cap

#np.savetxt("dCplant2018.csv", capa[419:][:], delimiter=",")
np.savetxt("dCsuelo2018_.csv", capa, delimiter=",")

cap.generate(grid_lat[419:],grid_lon[:],capa[419:][:],title,'viridis',1)
plt.show()
np.savetxt("grid_lat.csv", grid_lat[419:], delimiter=",")
np.savetxt("grid_lon.csv", grid_lon[:], delimiter=",")

