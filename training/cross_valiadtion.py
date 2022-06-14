import pandas as pd
import numpy as np
import lequi
from netCDF4 import Dataset
import mean_net


ISOTOPO = '15N'
n = 10
names_bio = ['bio1','bio2','bio3','bio4','bio5','bio6','bio7','bio8','bio9','bio10','bio11','bio12','bio13','bio14','bio15','bio16','bio17','bio18','bio19']


data = pd.read_csv("data/suelosAllAC.csv")
lon = np.array(data["lon"])
lat = np.array(data["lat"])
iso = np.array(data[ISOTOPO])


lon, lat, iso = lequi.csv(lon),lequi.csv(lat),lequi.csv(iso)

cluster = []
for i in lat:
    if i>=-25:
        cluster.append(0)
    elif i<=-29:
        cluster.append(1)


def call_var(name, n):
    dataset = Dataset('C:/Users/bmati/Desktop/proyecto296/training/data/biocliamtic_CR2_1979_2016/'+name+'.nc', mode="r")
    z = dataset.variables[name][:]
    grid_lat = dataset.variables['lat'][:]
    grid_lon = dataset.variables['lon'][:]
    dataset.close()
    return z[-n:], grid_lat, grid_lon

def significance(pvalue):
    if pvalue < 0.001:
        return ' ***'
    elif pvalue < 0.01:
        return ' **'
    elif pvalue < 0.05:
        return ' *'
    else:
        return ' '

bio_layers = {}
for name in names_bio:
    layer, grid_lat, grid_lon = call_var(name, n)
    col = mean_net.returnDat(lat,lon,grid_lat,grid_lon,np.nanmean(layer, axis=0))
    bio_layers[name] = col

bio_layers['Latitude'] = lat
bio_layers['Longitude'] = lon
bio_layers[ISOTOPO] = iso
bio_layers['cluster'] = cluster





df = pd.DataFrame(bio_layers)

names_bio.append('cluster')


from sklearn.model_selection import train_test_split
from sklearn import svm

X=[]
for key in names_bio:
    X.append(bio_layers[key])
X = np.array(X)
X = np.transpose(X)
print(X)
y = bio_layers[ISOTOPO]
y = np.array(y)



print(X.shape)
print(y.shape)

from sklearn.model_selection import cross_val_score
clf = svm.LinearSVR(C=1, random_state=42)
scores = cross_val_score(clf, X, y, cv=5)
print(scores)