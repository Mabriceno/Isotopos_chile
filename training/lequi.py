'''
Funciones varias

'''





#import matplotlib.pyplot as plt
import numpy as np

#elimina los valores fuera del area necesaria en la capa (z) generada por krige. 
#Para esto utiliza una capa (sst) previa como molde (por ejemplo rainfall o temp)
def shape(sst,z):
    n_y=[]
    for i in range(0,len(sst)):
        n_x=[]
        for j in range(0,len(sst[0])):
            if isinstance(sst[i][j], np.float32):
                n_x.append(z[i][j])
            else:
                n_x.append(sst[0][0])
        n_y.append(n_x)
    return n_y

#soluciona el cambio de "," a "." en valores tipo string que vienen de un archivo 
#csv para convertirlos en float
def csv(arr):
    x=[]
    for i in arr:
        n=""
        for j in i:
            if j!=",":
                n+=j
            else:
                n+="."
        x.append(float(n))
    return np.array(x)