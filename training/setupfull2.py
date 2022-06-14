import pandas as pd
from sklearn import linear_model
from netCDF4 import Dataset
import numpy as np
import lequi
import mean_net
import tensorflow as tf
import random
from keras.models import Sequential
from keras.layers import Dense
from scipy.stats import linregress
from sklearn.metrics import mean_squared_error


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

names_bio.append('cluster')

def setup(df, seed_x, names_bio, n_train):
    ################################################ SEPARAR TRAIN Y TEST DATA #################################################################
    ind = list(range(0,len(df['bio1']))) ##crea una lista de los indices para luego separarlos en train y test
    random.seed(seed_x)
    random.shuffle(ind)
    i = n_train ##numero de train data
    test_X_index = ind[:i]
    train_X_index = ind[i:] 
    df_train=pd.DataFrame(columns=list(df.keys()))
    df_test=pd.DataFrame(columns=list(df.keys()))
    for i in range(0,len(test_X_index)):
        df_test.loc[i] =  np.array(df.loc[test_X_index[i]])
    for i in range(0,len(train_X_index)):
        df_train.loc[i] =  np.array(df.loc[train_X_index[i]])


    ####MODEL 1
    r2 = 0
    for i in names_bio:
        for j in names_bio:
            for k in names_bio:
                model = linear_model.LinearRegression()
                X = df_train[[i,j,k,'cluster']]
                y = df_train[ISOTOPO]
                model.fit(X, y)
                r2_aux = model.score(X, y)
                if (i!=k and i!=j and j!=k) and r2_aux>r2:
                    r2 = r2_aux
                    model1 = model
                    list_model1 = [i,j,k]


    ####MODEL 2
    r2 = 0
    for i in names_bio:
        for j in names_bio:
                model = linear_model.LinearRegression()
                X = df_train[[i,j,'cluster']]
                y = df_train[ISOTOPO]
                model.fit(X, y)
                r2_aux = model.score(X, y)
                if (i!=j) and r2_aux>r2:
                    r2 = r2_aux
                    model2 = model
                    list_model2 = [i,j]


    ####model 3
    # split into input (X) and output (y) variables
    X = df_train[names_bio]
    y = df_train[ISOTOPO]

    #'sigmoid' 'tanh' 'linear'
    model = Sequential([
        Dense(128, input_dim=20, activation='tanh'),
        Dense(1280, activation='sigmoid'),
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


    ############PLOT MODEL 1
    x= df_test[ISOTOPO]
    y = []
    list_model1.append('cluster')
    for i in range(0, len(df_test['bio1'])):
        row = []
        for name in list_model1:
            row.append(list(df_test[name])[i])     
        y.append(list(model1.predict([row]))[0])

    cc = linregress(x,y)
    R1 = round(cc[2],2)
    S1 = significance(cc[3])


    ############PLOT MODEL 2
    x= df_test[ISOTOPO]
    y = []
    list_model2.append('cluster')
    for i in range(0, len(df_test['bio1'])):
        row = []
        for name in list_model2:
            row.append(list(df_test[name])[i])     
        y.append(list(model2.predict([row]))[0])

    cc = linregress(x,y)
    R2 = round(cc[2],2)
    S2 = significance(cc[3])


    ####plot model 3

    x = df_test[ISOTOPO]
    y_ = model.predict(df_test[names_bio])
    y= []
    for i in y_:
        y.append(i[0])

    cc = linregress(x,y)

    R3 = round(cc[2],2)
    S3 = significance(cc[3])


    return R1,R2,R3,S1,S2,S3


veces = 70
list_train = [15]

R1_list,R2_list,R3_list,S1_list,S2_list,S3_list = [],[],[],[],[],[]
seed_list = []
n_train_list = []
import sys 
for n_train in list_train:
    for i in range(0,veces):
        seed_x = random.randrange(sys.maxsize)
        print((df, seed_x, names_bio, n_train))
        R1,R2,R3,S1,S2,S3 = setup(df, seed_x, names_bio, n_train)
        R1_list.append(R1)
        R2_list.append(R2)
        R3_list.append(R3)
        S1_list.append(S1)
        S2_list.append(S2)
        S3_list.append(S3)
        seed_list.append(seed_x)
        n_train_list.append(n_train)

df= pd.DataFrame({
                    'seed':seed_list,
                    'R1': R1_list,
                    'R2': R2_list,
                    'R3': R3_list,
                    'S1': S1_list,
                    'S2': S2_list,
                    'S3': S3_list,
                    'n_train': n_train_list   })

df.to_csv('train_test.csv')


        