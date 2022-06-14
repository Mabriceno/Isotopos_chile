import plotly.graph_objects as go
from netCDF4 import Dataset
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots

# Load data
'''df = pd.read_csv(
    "https://raw.githubusercontent.com/plotly/datasets/master/finance-charts-apple.csv")
df.columns = [col.replace("AAPL.", "") for col in df.columns]
'''


def plot(fig, name, rcp, n_row, legend):

    def call_var(name):
        dataset = Dataset('data/'+rcp+name+'.nc', mode="r")
        z = dataset.variables[name][:]
        print(np.shape(z))
        #print(dataset.variables[name])
        #print(dataset.variables['time'])
        grid_lat = dataset.variables['lat'][:]
        grid_lon = []# dataset.variables['lon'][:]

        dataset.close()
        return z, grid_lat, grid_lon

    z, grid_lat, grid_lon = call_var(name)

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
        #print(error)
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
            
        ),
        go.Scatter(
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
        )]
    )
    # Set title



    for t in fig2.data:
        fig.append_trace(t, row=n_row, col=1)

    return fig
#fig.show()


    # Add range slider
'''    fig2.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1,
                        label="1m",
                        step="month",
                        stepmode="backward"),
                    dict(count=6,
                        label="6m",
                        step="month",
                        stepmode="backward"),
                    dict(count=1,
                        label="YTD",
                        step="year",
                        stepmode="todate"),
                    dict(count=1,
                        label="1y",
                        step="year",
                        stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(
                visible=True
            ),
            type="date"
        )
    )'''
