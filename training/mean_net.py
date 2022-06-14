import matplotlib.pyplot as plt
from netCDF4 import Dataset 
import numpy as np
import pandas as pd



##recibe una lista de numeros y un numero. Retorna la posicion del numero en la lista mas
#cercano al numero entregado.
def closest(lst, K): 
    n = lst[min(range(len(lst)), key = lambda i: abs(lst[i]-K))] 
    lst=list(lst)
    return lst.index(n)

def find_grid(lat,lon,grid_lat,grid_lon): 
    return closest(grid_lat, lat),closest(grid_lon, lon)

def returnDat(lat,lon,grid_lat,grid_lon,var):
    z=[]
    for i in range(0,len(lat)):
        x,y=find_grid(lat[i],lon[i],grid_lat,grid_lon)
        z.append(var[x][y])
    return np.array(z)
#b=find_grid(-35.97034603,-70.56827847, lat, lon)
#print(b)