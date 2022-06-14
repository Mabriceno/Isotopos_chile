import pandas as pd
from sklearn import linear_model
from netCDF4 import Dataset
import numpy as np
import lequi
import mean_net
import tensorflow as tf


ISOTOPO = '15N'
n = 10
names_bio = ['bio1','bio2','bio3','bio4','bio5','bio6','bio7','bio8','bio9','bio10','bio11','bio12','bio13','bio14','bio15','bio16','bio17','bio18','bio19']


data = pd.read_csv("data/suelosAllAC.csv")
lon = np.array(data["lon"])
lat = np.array(data["lat"])
iso = np.array(data[ISOTOPO])


lon, lat, iso = lequi.csv(lon),lequi.csv(lat),lequi.csv(iso)

