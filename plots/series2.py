import plotly.graph_objects as go
from netCDF4 import Dataset
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots

names_bio = ['bio1','bio2','bio3','bio4','bio5','bio6','bio7','bio8','bio9','bio10','bio11','bio12','bio13','bio14','bio15','bio16','bio17','bio18','bio19']

##sur: <-29 lat[0:78]
##norte: >-25 lat[122:len(lat)]

def call_var(name,rcp):
    dataset = Dataset('data/'+rcp+'/'+name+'.nc', mode="r")
    z = dataset.variables[name][:]
    grid_lat = dataset.variables['lat'][:]
    grid_lon = []# dataset.variables['lon'][:]
    dataset.close()
    return z, grid_lat, grid_lon


def mean_last_n(series, n):
    list_n = []
    x = []
    #for i in range(0,n):
    #    x.append(np.nan)


    for i in series:
        list_n.append(i)
        if len(list_n) == n:
            k = np.nanmean(list_n)
            x.append(k)
            list_n = list_n[1:len(list_n)]
    return x


def plot(fig, name, n_row, cluster):

    z_85, grid_lat, grid_lon = call_var(name, '85')
    z_26, grid_lat, grid_lon = call_var(name, '26')


    if cluster == 'Norte':
        z_85 = z_85[:,0:78,:]
        z_26 = z_26[:,0:78,:]
    else:
        z_85 = z_85[:,122:len(z_85[0]),:]
        z_26 = z_26[:,122:len(z_26[0]),:]

    time = []
    values_85 = []
    values_26 = []
    year = 2007
    error_up = []
    error_low = []

    for n in range(0, len(z_85)):
        value_85 = np.nanmean(z_85[n][:])
        error_85 = np.nanstd(z_85[n][:])
        values_85.append(value_85)
        time.append(year+n)
        error_up.append(value_85 + error_85)
        error_low.append(value_85 - error_85)
    for n in range(0, len(z_26)):
        value_26 = np.nanmean(z_26[n][:])
        #error_85 = np.nanstd(z_85[n][:])
        values_26.append(value_26)

    ventana_t = 10
    values_26 = mean_last_n(values_26, ventana_t)
    values_85= mean_last_n(values_85,ventana_t)

    values_26 = np.array(values_26)
    values_85 = np.array(values_85)
    time = np.array(time[ventana_t:])

    # Create figure
    fig2 = go.Figure([
        go.Scatter(
            x=list(time),
            y=list(values_26),
            name = 'RCP 2.6'+' '+cluster
            
        ),
        go.Scatter(
            x=list(time),
            y=list(values_85),
            name = 'RCP 8.5'+' '+cluster
            
        )
        ])

    
    # Set title
    
    for t in fig2.data:
        fig.append_trace(t, row=n_row, col=1)
    return fig



n_row = 1

fig = make_subplots(rows=2, cols=1)

ISOTOPO = 'N_Plant'


fig = plot(fig, ISOTOPO, 1, 'Norte')
fig = plot(fig, ISOTOPO, 2, 'Sur')

fig.show()


















'''
rcp = '85'
n_row = 1

fig = make_subplots(rows=3, cols=1)

for name in ['bio1','bio12']:
    fig = plot(fig, name, rcp, n_row, name)
    n_row += 1

fig.show()'''