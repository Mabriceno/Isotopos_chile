import pandas as pd

NAME = 'N_Soil26'
df = pd.read_csv(NAME+'.csv')



'''
import plotly.express as px
fig = px.density_mapbox(df, lat='lat', lon='lon', z='N_Soil', 
                        mapbox_style="stamen-terrain")
fig.show()'''

dff = df[df['date'] == 2007]
print(dff)

import plotly.express as px
import plotly.graph_objects as go

'''fig = px.scatter(dff, x="lat", y="lon", color="N_Soil",
                 title="Numeric 'size' values mean continuous color")'''

fig  = go.Figure(data=go.Scatter( x=dff["lat"], 
                                  y=dff["lon"],
                                  marker_color=dff["N_Soil"],
                                  mode='markers',
                                  marker_symbol= 'square'))
fig.show()