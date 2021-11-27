#Colm Keyes
#18/11/21
#Some basic college coding - beginning geoprocessing in Python

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
i=0

#Might have to use Rasterio to read in these image datasets
# Actually, looks like i might want to use GDAl to read these .img files, looks like erdas imagine docs point to that :)


image  = gdal.Open("C:\\Users\\Colm The Creator\\Documents\\Data\\RemoteSensing_Data\\nop.img")
#image_gtiff = gdal.Translate('C:\\Users\\Colm The Creator\\Documents\\Data\\RemoteSensing_Data\\nop.tif',image ,format="GTiff")

#nop_gtiff = rs.open('C:\\Users\\Colm The Creator\\Documents\\Data\\RemoteSensing_Data\\nop.tif')

#rsplt.show(nop_gtiff)
##############################
#Kmeans unsupervised classification using Sklearn, for 1 band
##############################
band = image.GetRasterBand(1)
gdal_band = band.ReadAsArray()
gdal_img_flat = gdal_band.reshape((-1,1))
kmeans = KMeans(10)
kmeans.fit(gdal_img_flat)
kmeans_fit_result = kmeans.labels_
kmeans_cluster = kmeans_fit_result.reshape(gdal_band.shape)
plt.figure(figsize=(20,20))
plt.title("Unsupervised KMeans Classification for Band 1")
plt.xlabel("DN")
plt.ylabel("DN")
plt.legend()
plt.imshow(kmeans_cluster, cmap="hsv")
#plt.show()


##############################
#Kmeans unsupervised classification using Sklearn, for all bands
##############################
#We set up a blank array with the raster shape of (500,480,6)

empty_raster_shape = np.zeros((image.RasterYSize,image.RasterXSize,image.RasterCount))


for b in range(empty_raster_shape.shape[2]):
    empty_raster_shape[:, :, b] = image.GetRasterBand(b + 1).ReadAsArray()

new_shape = (empty_raster_shape.shape[0] * empty_raster_shape.shape[1], empty_raster_shape.shape[2])

X = empty_raster_shape[:, :, :6].reshape(new_shape)

k_means = KMeans(n_clusters=8)
k_means.fit(X)

X_cluster = k_means.labels_
X_cluster = X_cluster.reshape(empty_raster_shape[:, :, 0].shape)

plt.figure(figsize=(20,20))
plt.imshow(X_cluster, cmap="hsv")
plt.title("Unsupervised KMeans Classification for All Bands Combined")
plt.xlabel("DN")
plt.ylabel("DN")
plt.legend()
plt.show()