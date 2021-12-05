#Colm Keyes
#18/11/21
#Some basic college coding - Section 11 some Unsupervised land classification

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

image  = gdal.Open("C:\\Users\\Colm The Creator\\Documents\\Data\\RemoteSensing_Data\\image_derived_spectral_mapping.img")

##############################
#Kmeans unsupervised classification using Sklearn, for 1 band
##############################
band_1 = image.GetRasterBand(1)
band_2 = image.GetRasterBand(2)
band_3 = image.GetRasterBand(3)
#f, (sub1, sub2, sub3) = plt.subplots(1, 3)

for band in [band_1,band_2,band_3]:
    gdal_band = band.ReadAsArray()
    gdal_img_flat = gdal_band.reshape((-1,1))
    kmeans = KMeans(10)
    kmeans.fit(gdal_img_flat)
    kmeans_fit_result = kmeans.labels_
    kmeans_cluster = kmeans_fit_result.reshape(gdal_band.shape)
    #plt.figure(figsize=(20,20))
    plt.title("Unsupervised KMeans Classification for Band"+str(band.GetBand()))
    plt.legend()
    plt.imshow(kmeans_cluster, cmap="hsv")
    plt.show()


#plt.show()


##############################
#Kmeans unsupervised classification using Sklearn, for all bands
##############################
#We set up a blank array with the raster shape of (500,480,6)

empty_raster_shape = np.zeros((image.RasterYSize,image.RasterXSize,image.RasterCount))


for b in range(empty_raster_shape.shape[2]):
    empty_raster_shape[:, :, b] = image.GetRasterBand(b + 1).ReadAsArray()

new_shape = (empty_raster_shape.shape[0] * empty_raster_shape.shape[1], empty_raster_shape.shape[2])

X = empty_raster_shape[:, :, :image.RasterCount].reshape(new_shape)

k_means = KMeans(n_clusters=10)
k_means.fit(X)

X_cluster = k_means.labels_
X_cluster = X_cluster.reshape(empty_raster_shape[:, :, 0].shape)

plt.figure(figsize=(20,20))
plt.imshow(X_cluster, cmap="hsv")
plt.title("Unsupervised KMeans Classification for All Bands Combined")
plt.xlabel()
plt.ylabel()
plt.legend()
plt.show()


















