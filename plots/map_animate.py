import plotly.graph_objects as go
import pandas as pd


NAME = 'N_Soil26'
df = pd.read_csv(NAME+'.csv')

n_year = 2007
dff = df[df['date'] == n_year]


def select_year(year,df):
    dff = df[df['date'] == year]
    return dff

fig = go.Figure(
    data=go.Scatter( x=dff["lat"], 
                    y=dff["lon"],
                    marker_color=dff["N_Soil"],
                    mode='markers',
                    marker_symbol= 'square'),

    layout=go.Layout(
        title="Start Title",
        updatemenus=[dict(
            type="buttons",
            buttons=[dict(label="Play",
                          method="animate",
                          args=[None])])]
    ),
    frames=[go.Frame(
            data=go.Scatter( 
                    x=select_year(n_year, df)["lat"], 
                    y=select_year(n_year, df)["lon"],
                    marker_color=select_year(n_year, df)["N_Soil"],
                    mode='markers',
                    marker_symbol= 'square'))

            for n_year in range(2008,2050)]
)

fig.show()