#Colm Keyes
#30/11/21
#Some basic college coding - beginning geoprocessing in Python (FAILED, matplotlib has difficulty dealing with plotting
# the featurespace

########################
#Imports & Inits
########################
import geopandas as gpd
import os, sys

import matplotlib.pyplot as plt
import osgeo
from osgeo import gdal
import rasterio as rs
import rasterio.plot as rsplt
from sklearn.cluster import KMeans
import numpy as np
import random
import pandas as pd
from itertools import chain
i=0

#Might have to use Rasterio to read in these image datasets
# Actually, looks like i might want to use GDAl to read these .img files, looks like erdas imagine docs point to that :)


#image  = gdal.Open("C:\\Users\\Colm The Creator\\Documents\\Data\\RemoteSensing_Data\\rosis_alora.img")

image = gdal.Open("C:\\Users\\Colm The Creator\\Documents\\Data\\RemoteSensing_Data\\t0_r0_stacked.img")


temp = image.GetRasterBand(1).ReadAsArray().astype("int16")           #= image.Read(1)
albedo = image.GetRasterBand(2).ReadAsArray().astype("float")          #image.read(2, output_shape=(1,len(temp)))

temp_flat = temp.reshape((-1,1))
temp_list = list(chain.from_iterable(temp_flat))


albedo_flat = albedo.reshape((-1,1))
albedo_list = list(chain.from_iterable(albedo_flat))

plt.scatter(temp_list,albedo_list)
plt.show()
# temp_list = temp_flat.tolist()
#
# rand_samp_temp = random.sample(temp_list,1000)
#
# rand_samp_temp = random.sample(albedo_list,1000)


# plt.scatter(rand_samp_temp,rand_samp_temp)
# plt.show()
