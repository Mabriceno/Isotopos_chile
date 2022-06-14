import pandas as pd
from sklearn import linear_model
from netCDF4 import Dataset
import numpy as np
import lequi
import mean_net
import tensorflow as tf


ISOTOPO = '15N'
n = 10
names_bio = ['bio1','bio2','bio3','bio4','bio5','bio6','bio7','bio8','bio9','bio10','bio11','bio12','bio13','bio14','bio15','bio16','bio17','bio18','bio19']


data = pd.read_csv("data/suelosAllAC.csv")
lon = np.array(data["lon"])
lat = np.array(data["lat"])
iso = np.array(data[ISOTOPO])


lon, lat, iso = lequi.csv(lon),lequi.csv(lat),lequi.csv(iso)

cluster = []
for i in lon:
    if i>=-25:
        cluster.append(0)
    elif i<=29:
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
print(df)

################################################ SEPARAR TRAIN Y TEST DATA #################################################################

ind = list(range(0,len(df['bio1']))) ##crea una lista de los indices para luego separarlos en train y test

import random
#random.seed(1)
random.shuffle(ind)

i = 13 ##numero de train data
test_X_index = ind[:i]
train_X_index = ind[i:] 

df_train=pd.DataFrame(columns=list(df.keys()))
df_test=pd.DataFrame(columns=list(df.keys()))


for i in range(0,len(test_X_index)):
    df_test.loc[i] =  np.array(df.loc[test_X_index[i]])

for i in range(0,len(train_X_index)):
    df_train.loc[i] =  np.array(df.loc[train_X_index[i]])


####MODEL 1

from sklearn.metrics import mean_squared_error

random.shuffle(names_bio)
mse = 0
for i in names_bio:
    for j in names_bio:
        for k in names_bio:
            model = linear_model.LinearRegression()
            X = df_train[[i,j,k,'cluster']]
            y = df_train[ISOTOPO]
            model.fit(X, y)
            mse_aux = mean_squared_error( y, model.predict(X) )
            if (i!=k and i!=j and j!=k) and mse_aux>mse:
                print([i,j,k])
                mse = mse_aux
                model1 = model
                list_model1 = [i,j,k]

print(list_model1)

####MODEL 4


mse = 0
for i in names_bio:
    for j in names_bio:
            model = linear_model.LinearRegression()
            X = df_train[[i,j,'cluster']]
            y = df_train[ISOTOPO]
            model.fit(X, y)
            mse_aux = mean_squared_error( y, model.predict(X) )
            if (i!=j) and mse_aux>mse:
                print([i,j])
                mse = mse_aux
                model4 = model
                list_model4 = [i,j]

print(list_model4)

##MODEL 2 (MULTI LINEAR MODEL)
names_bio.append('cluster')

X = df_train[names_bio]
y = df_train[ISOTOPO]

model2 = linear_model.LinearRegression()
model2.fit(X, y)


####model 3
import keras
print(keras.__version__)
from keras.models import Sequential
from keras.layers import Dense

'''y = np.array(df_train[ISOTOPO])
print(y)
del df_train['OC']
X=[]
for i in range(0,len(df_train['bio1'])):
    X.append(df_train.loc[i])'''
# split into input (X) and output (y) variables
X = df_train[names_bio]
y = df_train[ISOTOPO]

# define the keras model

#'sigmoid' 'tanh' 'linear'
model = Sequential([
    Dense(128, input_dim=20, activation='sigmoid'),
    Dense(1280, activation='relu'),
    Dense(1280, activation='relu'),
    Dense(1280, activation='relu'),
    Dense(1280, activation='relu'),
    Dense(1700, activation='relu'),
    Dense(128, activation='relu'),
    Dense(128, activation='relu'),
    Dense(128, activation='relu'),
    Dense(128, activation='relu'),
    Dense(80, activation='relu'),
    Dense(62, activation='relu'),
    Dense(1, activation='linear')])

# compile the keras model
model.compile(loss='mean_squared_error', optimizer='adam', metrics=[tf.keras.metrics.MeanAbsoluteError()])

# fit the keras model on the dataset
history = model.fit(X, y, epochs=110, batch_size=len(X))

import matplotlib.pyplot as plt
from scipy.stats import linregress

# summarize history for accuracy
#plt.plot(history.history['mean_squared_error'])
plt.plot(history.history['mean_absolute_error'])
#plt.plot(history.history['mean_absolute_percentage_error'])
#plt.plot(history.history['cosine_proximity'])
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()





import matplotlib.pyplot as plt
from scipy.stats import linregress


############PLOT MODEL 1
ax = plt.subplot(1,1,1).set_title('M1 \n Multilineal Model')
x= df_test[ISOTOPO]
y = []

list_model1.append('cluster')
for i in range(0, len(df_test['bio1'])):
    row = []
    for name in list_model1:
        row.append(list(df_test[name])[i])     
    y.append(list(model1.predict([row]))[0])

cc = linregress(x,y)
plt.scatter(x,y,  color='black')
line,=plt.plot(x,x*cc[0]+cc[1], color='grey')
line.set_label('$R = $'+ str(round(cc[2],2))+ significance(cc[3]))
plt.xlabel(' Test Data ')
plt.ylabel(' Predicted Data ')
plt.legend()
plt.show()

############PLOT MODEL 4
ax = plt.subplot(1,1,1).set_title('M4 \n Multilineal Model')
x= df_test[ISOTOPO]
y = []

list_model4.append('cluster')
for i in range(0, len(df_test['bio1'])):
    row = []
    for name in list_model4:
        row.append(list(df_test[name])[i])     
    y.append(list(model4.predict([row]))[0])

cc = linregress(x,y)
plt.scatter(x,y,  color='black')
line,=plt.plot(x,x*cc[0]+cc[1], color='grey')
line.set_label('$R = $'+ str(round(cc[2],2))+ significance(cc[3]))
plt.xlabel(' Test Data ')
plt.ylabel(' Predicted Data ')
plt.legend()
plt.show()



############PLOT MODEL 2
ax = plt.subplot(1,1,1).set_title('M2 \n Multilineal Model')
x= df_test[ISOTOPO]
y = []


'''for i in range(0, len(df_test['bio1'])):
    row = []
    for name in names_bio:
        row.append(list(df_test[name])[i])     
    y.append(list(model2.predict([row]))[0])'''

y = model2.predict(df_test[names_bio])
cc = linregress(x,y)
plt.scatter(x,y,  color='black')
line,=plt.plot(x,x*cc[0]+cc[1], color='grey')
line.set_label('$R = $'+ str(round(cc[2],2))+ significance(cc[3]))
plt.xlabel(' Test Data ')
plt.ylabel(' Predicted Data ')
plt.legend()



plt.show()

####plot model 3

'''x = np.array(list(df_test[ISOTOPO]))

del df_test[ISOTOPO]
X=[]
for i in range(0,len(df_test['bio1'])):
    X.append(df_test.loc[i])
X = np.array(X)
predictions = model.predict(X)
y=[]
for i in predictions:
    y.append(i[0])

y=np.array(y)
'''
x = df_test[ISOTOPO]
y_ = model.predict(df_test[names_bio])
y= []
for i in y_:
    y.append(i[0])

print((x,y))

ax = plt.subplot(1,1,1).set_title('M3 \n MLP')
cc = linregress(x,y)
plt.scatter(x,y,  color='black')
line,=plt.plot(x,x*cc[0]+cc[1], color='grey')
line.set_label('$R = $'+ str(round(cc[2],2))+ significance(cc[3]))
plt.xlabel(' Test Data ')
plt.ylabel(' Predicted Data ')
plt.legend()
plt.show()
