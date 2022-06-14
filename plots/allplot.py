import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np
from netCDF4 import Dataset
import histo
import series


name1 = 'C_Plant'
name2 = 'C_Plant'
rcp26 = '26/'
rcp85 = '85/'


fig = make_subplots(rows=2, cols=2)
fig = histo.plot(fig, name1,rcp26, 1)
fig = histo.plot(fig, name1,rcp85, 2)
fig = series.plot(fig, name1,rcp26,1, '$\delta N Plant $')
fig = series.plot(fig, name1,rcp85,2, '$\delta C Plant $')

fig.update_layout(
                  title_text=r'$\text{Mean }\delta N \text{ and } \delta C \text{ Plant prediction using a low emissions scenario (RCP 2.6)}$')

fig.show()
