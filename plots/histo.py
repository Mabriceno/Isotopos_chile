import plotly.figure_factory as ff
import numpy as np
import plotly.graph_objects as go
from netCDF4 import Dataset

# Add histogram data
'''x1 = np.random.randn(200) - 2
x2 = np.random.randn(200)
x3 = np.random.randn(200) + 2
x4 = np.random.randn(200) + 4'''

def plot(fig, name,rcp, n_row):
    def call_var(name):
        dataset = Dataset('data/'+rcp+name+'.nc', mode="r")
        z = dataset.variables[name][:]
        #print(dataset.variables[name])
        #print(dataset.variables['time'])
        grid_lat = dataset.variables['lat'][:]
        grid_lon = dataset.variables['lon'][:]

        dataset.close()
        return z, grid_lat, grid_lon

    z, grid_lat, grid_lon = call_var(name)

    def get_layer_year(cubo, i_year):
        z=[]
        for lat in cubo[i_year]:
            for lon in lat:
                if isinstance(lon, float):
                    z.append(lon)
        z = np.array(z)
        z = z[np.logical_not(np.isnan(z))]
        return z

    x1 = get_layer_year(z, 13)
    x2 = get_layer_year(z, 23)
    x3 = get_layer_year(z, 33)
    x4 = get_layer_year(z, 43)

    print(x1)
    # Group data together
    hist_data = [x1, x2, x3, x4]

    group_labels = ['2020', '2030', '2040', '2050']

    # Create distplot with custom bin_size
    fig2 = ff.create_distplot(hist_data, group_labels) #bin_size=.2)
    #fig2 = ff.create_distplot(hist_data, group_labels)


    ##hitogramas 1
    fig.add_trace(go.Histogram(fig2['data'][0],
                           marker_color='#68BBE3',
                          ), row=n_row, col=2),
    fig.add_trace(go.Histogram(fig2['data'][1],
                           marker_color='#0E86D4'
                          ), row=n_row, col=2),
    fig.add_trace(go.Histogram(fig2['data'][2],
                           marker_color='#055C9D'
                          ), row=n_row, col=2),
    fig.add_trace(go.Histogram(fig2['data'][3],
                           marker_color='#003060'
                          ), row=n_row, col=2),
    
    ##histogramas 2

    fig.add_trace(go.Scatter(fig2['data'][4],
                         line=dict(color='#68BBE3', width=0.5)
                        ), row=n_row, col=2)

    fig.add_trace(go.Scatter(fig2['data'][5],
                         line=dict(color='#0E86D4', width=0.5)
                        ), row=n_row, col=2)

    fig.add_trace(go.Scatter(fig2['data'][6],
                         line=dict(color='#055C9D', width=0.5)
                        ), row=n_row, col=2)

    fig.add_trace(go.Scatter(fig2['data'][7],
                         line=dict(color='#003060', width=0.5)
                        ), row=n_row, col=2)

    ## distribucion

    # rug / margin plot to immitate ff.create_distplot

    # some manual adjustments on the rugplot
    #fig.update_yaxes(range=[0.95,1.15], tickfont=dict(color='rgba(0,0,0,0)', size=14), row=2, col=2)
    #fig.update_layout(showlegend=False)                
    
    #fig.show()

    return fig

''' 
rug1 = np.repeat(1,len(x1))
    rug2 = np.repeat(2,len(x1))
    rug3 = np.repeat(3,len(x1))
    rug4 = np.repeat(4,len(x1))

    fig.add_trace(go.Scatter(x=x1, y = rug1,
                        mode = 'markers',
                        marker=dict(color = 'blue', symbol='line-ns-open')
                            ), row=2, col=2)

    fig.add_trace(go.Scatter(x=x2, y = rug2,
                        mode = 'markers',
                        marker=dict(color = 'red', symbol='line-ns-open')
                            ), row=2, col=2)
    fig.add_trace(go.Scatter(x=x3, y = rug3,
                        mode = 'markers',
                        marker=dict(color = 'green', symbol='line-ns-open')
                            ), row=2, col=2)

    fig.add_trace(go.Scatter(x=x4, y = rug4,
                        mode = 'markers',
                        marker=dict(color = 'yellow', symbol='line-ns-open')
                            ), row=2, col=2)
'''