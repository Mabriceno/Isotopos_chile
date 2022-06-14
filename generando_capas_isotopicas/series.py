import plotly.graph_objects as go
from netCDF4 import Dataset
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots

names_bio = ['bio1','bio2','bio3','bio4','bio5','bio6','bio7','bio8','bio9','bio10','bio11','bio12','bio13','bio14','bio15','bio16','bio17','bio18','bio19']


def call_var(name,rcp):
    dataset = Dataset('C:/Users/bmati/Desktop/proyecto296/generando_capas_isotopicas/Data/bioclimatic_Reg'+rcp+'_2007_2050/'+name+'.nc', mode="r")
    z = dataset.variables[name][:]
    grid_lat = dataset.variables['lat'][:]
    grid_lon = []# dataset.variables['lon'][:]
    dataset.close()
    return z, grid_lat, grid_lon

def plot(fig, name, rcp, n_row, legend):

    z, grid_lat, grid_lon = call_var(name,rcp)

    time = []
    values = []
    year = 2007
    error_up = []
    error_low = []

    for n in range(0, len(z)):
        value = np.nanmean(z[n][:])
        error = np.nanstd(z[n][:])
        values.append(value)
        time.append(year+n)
        error_up.append(value + error)
        error_low.append(value - error)


    values = np.array(values)
    time = np.array(time)

    # Create figure
    fig2 = go.Figure([
        go.Scatter(
            x=list(time),
            y=list(values),
            name = legend
            
        )])

    
    # Set title

    for t in fig2.data:
        fig.append_trace(t, row=n_row, col=1)

    return fig


rcp = '85'
n_row = 1

fig = make_subplots(rows=3, cols=1)

for name in ['bio1','bio12']:
    fig = plot(fig, name, rcp, n_row, name)
    n_row += 1

fig.show()


''' go.Scatter(
            #name='Upper Bound',
            x=list(time),
            y=error_up,
            mode='lines',
            marker=dict(color="#444"),
            line=dict(width=0),
            showlegend=False
        ),
        go.Scatter(
            #name='Lower Bound',
            x=list(time),
            y=error_low,
            marker=dict(color="#444"),
            line=dict(width=0),
            mode='lines',
            fillcolor='rgba(68, 68, 68, 0.3)',
            fill='tonexty',
            showlegend=False
        )'''