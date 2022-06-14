import pickle
import pandas as pd
import numpy as np 
from bio import Param_bioclimaticos as Pb
from netCDF4 import Dataset
import lequi
import tensorflow as tf

################################################ CARGAR Y ORDENAR DATA ###############################################

####### LLama a .lec
filehandler = open('parametros_bioclimaticos.lec', 'rb') 
pb = pickle.load(filehandler)

######## Generate a grid:
'''dataset = Dataset("pr.nc", mode="r")
grid_lat = dataset.variables['lat'][:]
grid_lon = dataset.variables['lon'][:]
dataset.close()

##call csv
#data = pd.read_csv("Datos/suelosAllpromAC.csv")
data = pd.read_csv("Datos/CHLSOC_v1.0.csv", encoding = "ISO-8859-1")
print(len(data['oc']))
data = data.dropna(subset=['oc','long','lat','year'])
print(len(data['oc']))


lon = np.array(data["long"])
lat = np.array(data["lat"])
oc = np.array(data["oc"])
year = np.array(data["year"])

#lon, lat, oc = lequi.csv(lon),lequi.csv(lat),lequi.csv(oc)

##Generate dataFrame
import mean_net

df = pd.DataFrame({ 
                    'Latitude':lat,
                    'Longitude':lon,
                    'OC': oc,
                    'year':year,
                    'bio1':mean_net.returnDat(lat,lon,grid_lat,grid_lon,pb.bio1),
                    'bio2':mean_net.returnDat(lat,lon,grid_lat,grid_lon,pb.bio2),
                    'bio3':mean_net.returnDat(lat,lon,grid_lat,grid_lon,pb.bio3),
                    'bio4':mean_net.returnDat(lat,lon,grid_lat,grid_lon,pb.bio4),
                    'bio5':mean_net.returnDat(lat,lon,grid_lat,grid_lon,pb.bio5),
                    'bio6':mean_net.returnDat(lat,lon,grid_lat,grid_lon,pb.bio6),
                    'bio7':mean_net.returnDat(lat,lon,grid_lat,grid_lon,pb.bio7),
                    'bio8':mean_net.returnDat(lat,lon,grid_lat,grid_lon,pb.bio8),
                    'bio9':mean_net.returnDat(lat,lon,grid_lat,grid_lon,pb.bio9),
                    'bio10':mean_net.returnDat(lat,lon,grid_lat,grid_lon,pb.bio10),
                    'bio11':mean_net.returnDat(lat,lon,grid_lat,grid_lon,pb.bio11),
                    'bio12':mean_net.returnDat(lat,lon,grid_lat,grid_lon,pb.bio12),
                    'bio13':mean_net.returnDat(lat,lon,grid_lat,grid_lon,pb.bio13),
                    'bio14':mean_net.returnDat(lat,lon,grid_lat,grid_lon,pb.bio14),
                    'bio15':mean_net.returnDat(lat,lon,grid_lat,grid_lon,pb.bio15),
                    'bio16':mean_net.returnDat(lat,lon,grid_lat,grid_lon,pb.bio16),
                    'bio17':mean_net.returnDat(lat,lon,grid_lat,grid_lon,pb.bio17),
                    'bio18':mean_net.returnDat(lat,lon,grid_lat,grid_lon,pb.bio18),
                    'bio19':mean_net.returnDat(lat,lon,grid_lat,grid_lon,pb.bio19)})


df.to_csv('carbon_dataframe.csv')

print(ccc)'''
df = pd.read_csv("carbon_dataframe.csv")
print(len(df['OC']))
df = df.dropna()
print(len(df['OC']))

df = pd.DataFrame({ 
                    'Latitude':np.array(df['Latitude']),
                    'Longitude':np.array(df['Longitude']),
                    'OC': np.array(df['OC']),
                    'year': np.array(df['year']),
                    'bio1':np.array(df['bio1']),
                    'bio2':np.array(df['bio2']),
                    'bio3':np.array(df['bio3']),
                    'bio4':np.array(df['bio4']),
                    'bio5':np.array(df['bio5']),
                    'bio6':np.array(df['bio6']),
                    'bio7':np.array(df['bio7']),
                    'bio8':np.array(df['bio8']),
                    'bio9':np.array(df['bio9']),
                    'bio10':np.array(df['bio10']),
                    'bio11':np.array(df['bio11']),
                    'bio12':np.array(df['bio12']),
                    'bio13':np.array(df['bio13']),
                    'bio14':np.array(df['bio14']),
                    'bio15':np.array(df['bio15']),
                    'bio16':np.array(df['bio16']),
                    'bio17':np.array(df['bio17']),
                    'bio18':np.array(df['bio18']),
                    'bio19':np.array(df['bio19'])})

############################################################################################################################################
################################################ SEPARAR TRAIN Y TEST DATA #################################################################

ind = list(range(0,len(df['bio1']))) ##crea una lista de los indices para luego separarlos en train y test

import random
random.seed(1)
random.shuffle(ind)

i = 3300 ##numero de train data
test_X_index = ind[:i]
train_X_index = ind[i:] 

df_train=pd.DataFrame(columns=list(df.keys()))
df_test=pd.DataFrame(columns=list(df.keys()))


for i in range(0,len(test_X_index)):
    df_test.loc[i] =  np.array(df.loc[test_X_index[i]])

for i in range(0,len(train_X_index)):
    df_train.loc[i] =  np.array(df.loc[train_X_index[i]])



###############################################################################################################################################
############################################### ENTRENAR MODELOS ##############################################################################

ISOTOPO = 'OC' ##USAR N O C 
corr = df_train.corr()

'''############################################ Modelo 1, lineal simple con mejor R ####
R=0
indi=0
for i in range(3,len(corr[ISOTOPO])): ##3 para solo contar las variables bioclimaticas
    v=np.sqrt(corr[ISOTOPO][i]**2)
    if v>R:
        R=v
        indi=i

print(list(corr)[indi])

import statsmodels.formula.api as smf
model1 = smf.ols(ISOTOPO+' ~ '+list(corr)[indi], data=df_train).fit()

print('\n                      MODELO 1                            \n')
print(model1.summary())
print(model1.params[1])
############################################ Modelo 2, multilineal de tres variables con mejor R ####

variables=['Latitude', 'Longitude', 'bio1', 'bio2', 'bio3', 'bio4', 'bio5', 'bio6', 'bio7', 'bio8', 'bio9', 'bio10', 'bio11', 'bio12', 'bio13', 'bio14', 'bio15', 'bio16', 'bio17', 'bio18', 'bio19']

R2=0
for i in variables:
    for j in variables:
        for k in variables:
            formula=ISOTOPO+' ~ + '+i+' + '+j+' + '+k
            mod = smf.ols(formula, data=df_train).fit()
            if mod.rsquared>R2:
                R2=mod.rsquared
                form=formula
                A_=[i,j,k]

print('\n                      MODELO 2                            \n')
print(R2)
print(form)
model2 = smf.ols(form, data=df_train).fit()
print(model2.summary())
print(model2.params)
print(A_)'''


############################################ Modelo 3, deep learning ####
import keras
print(keras.__version__)
from keras.models import Sequential
from keras.layers import Dense

y = np.array(df_train[ISOTOPO])
print(y)
del df_train['OC']
X=[]
for i in range(0,len(df_train['bio1'])):
    X.append(df_train.loc[i])
# split into input (X) and output (y) variables
X = np.array(X)

# define the keras model

#'sigmoid' 'tanh' 'linear'
model = Sequential([
    Dense(128, input_dim=22, activation='relu'),
    Dense(1280, activation='relu'),
    Dense(1280, activation='relu'),
    Dense(128, activation='relu'),
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
history = model.fit(X, y, epochs=130, batch_size=len(X))

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

# evaluate the keras model
#_, accuracy = model.evaluate(X, y)
#print('Accuracy: %.2f' % (accuracy*100))


###############################################################################################################################################
############################################### TESTEAR MODELOS  ##############################################################################


def significance(pvalue):
    if pvalue < 0.001:
        return ' ***'
    elif pvalue < 0.01:
        return ' **'
    elif pvalue < 0.05:
        return ' *'
    else:
        return ' '


'''##MODEL 1
B=model1.params[0]
A=model1.params[1]
ax = plt.subplot(1,2,1).set_title('M1 \n Lineal Model')
x= df_test[ISOTOPO]
y= A*df_test[list(corr)[indi]] + B
mask = ~np.isnan(x) & ~np.isnan(y)

print('x :'+str(x))
print('y :'+str(y))

cc = linregress(x[mask],y[mask])
print('cc :'+str(cc))

plt.scatter(x,y,  color='black')
line,=plt.plot(x,x*cc[0]+cc[1], color='grey')
line.set_label('$R = $'+ str(round(cc[2],2))+ significance(cc[3]))
plt.xlabel(' Test Data ')
plt.ylabel(' Predicted Data ')
plt.legend()


##MODEL 2
B=model2.params[0]
A1=model2.params[1]
A2=model2.params[2]
A3=model2.params[3]
ax = plt.subplot(1,2,2).set_title('M2 \n Multilineal Model')
x= df_test[ISOTOPO]
y= A1*df_test[A_[0]] + A2*df_test[A_[1]] + A3*df_test[A_[2]] + B

mask = ~np.isnan(x) & ~np.isnan(y)

cc = linregress(x[mask],y[mask])
plt.scatter(x,y,  color='black')
line,=plt.plot(x,x*cc[0]+cc[1], color='grey')
line.set_label('$R = $'+ str(round(cc[2],2))+ significance(cc[3]))
plt.xlabel(' Test Data ')
plt.ylabel(' Predicted Data ')
plt.legend()

'''
##MODEL 3

x = np.array(list(df_test[ISOTOPO]))

del df_test['OC']
X=[]
for i in range(0,len(df_test['bio1'])):
    X.append(df_test.loc[i])
X = np.array(X)
predictions = model.predict(X)
y=[]
for i in predictions:
    y.append(i[0])

y=np.array(y)


print('x,y :')
print((len(x),len(y)))
ax = plt.subplot(1,1,1).set_title('M3 \n MLP')

cc = linregress(x,y)
plt.scatter(x,y,  color='black')
line,=plt.plot(x,x*cc[0]+cc[1], color='grey')
line.set_label('$R = $'+ str(round(cc[2],2))+ significance(cc[3]))
plt.xlabel(' Test Data ')
plt.ylabel(' Predicted Data ')
plt.legend()
plt.show()

