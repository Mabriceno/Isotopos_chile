import plotly.express as px
import pandas as pd
import numpy as np
import lequi
import matplotlib.pyplot as plt


data = pd.read_csv("data/plantasAll.csv")
lon_p = np.array(data["lon"])
lat_p = np.array(data["lat"])

data = pd.read_csv("data/suelosAllAC.csv")
lon_s = np.array(data["lon"])
lat_s = np.array(data["lat"])


lon_p, lat_p, = lequi.csv(lon_p),lequi.csv(lat_p)
lon_s, lat_s, = lequi.csv(lon_s),lequi.csv(lat_s)

muestra_p = []
for i in lon_p:
    muestra_p.append('green')

muestra_s = []
for i in lon_s:
    muestra_s.append('blue')

lon  = np.concatenate((lon_p, lon_s), axis = 0)
lat  = np.concatenate((lat_p, lat_s),  axis = 0)
muestra = np.concatenate((muestra_p, muestra_s), axis = 0)


df = pd.DataFrame({
                    'lat':lat,
                    'lon': lon,
                    'muestra': muestra})
'''
fig = px.scatter(x = lat, y = lon)
fig.show()'''

df.plot(x='lon', y='lat', kind='scatter', c='muestra', colormap='YlOrRd')
plt.show()