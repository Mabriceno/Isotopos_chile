import pandas as pd
import numpy as np

df = pd.read_csv('train_test.csv')

df10 = {}
df13 = {}
df15 = {}






def a(df, i, f, name):
    print(name)
    R1 = np.mean(np.array(df['R1'][i:f]))
    R2 = np.mean(np.array(df['R2'][i:f]))
    R3 = np.mean(np.array(df['R3'][i:f]))
    #R4 = np.mean(np.array(df['R4'][i:f]))

    S1 = np.std(np.array(df['R1'][i:f]))
    S2 = np.std(np.array(df['R2'][i:f]))
    S3 = np.std(np.array(df['R3'][i:f]))
    #S4 = np.std(np.array(df['R4'][i:f]))

    print('Model 1:'+ str(R1) + ' ' +str(S1))
    print('Model 2:'+ str(R2) + ' ' +str(S2))
    print('Model 3:'+ str(R3) + ' ' +str(S3))
    #print('Model 4:'+ str(R4) + ' ' +str(S4))

a(df,0,70, '15')

