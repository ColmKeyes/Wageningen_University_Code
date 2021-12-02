#Colm Keyes
#30/11/21
#Some basic college coding - calculating Redness & iron content for an image

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


#image  = gdal.Open("C:\\Users\\Colm The Creator\\Documents\\Data\\RemoteSensing_Data\\rosis_alora.img")

image = rs.open("C:\\Users\\Colm The Creator\\Documents\\Data\\RemoteSensing_Data\\rosis_alora.img")


red_band_range = list(np.arange(47,72))
green_band_range = list(np.arange(22,47))
blue_band_range = list(np.arange(1,22))


red_bands = image.read(red_band_range)
green_bands = image.read(green_band_range)
blue_bands = image.read(blue_band_range)

redness = red_bands.mean()/(red_bands.mean()+green_bands.mean()+blue_bands.mean())


fe_content = (redness-0.47)/0.0048




#################################################
#section 2
#################################################

image_corrected = rs.open("C:\\Users\\Colm The Creator\\Documents\\Data\\RemoteSensing_Data\\rosis_alora_cr.img")

bands_image_cr = image_corrected.read((12,55,85))

red_band = image_corrected.read(12)
green_band = image_corrected.read(55)
blue_band = image_corrected.read(85)

# Normalize bands into 0.0 - 1.0 scale
def normalize(array):
    array_min, array_max = array.min(), array.max()
    return (array - array_min) / (array_max - array_min)


red_norm = normalize(red_band)
green_norm = normalize(green_band)
blue_norm = normalize(blue_band)

redness_1 = red_norm/(red_norm+green_norm+blue_norm)
fe_content_1 = (redness_1-0.47)/0.0048


rgb_stack = np.dstack((red_norm, green_norm, blue_norm))

f,(sub1,sub2,sub3) = plt.subplots(1,3)

sub1.title.set_text("Redness")
sub2.title.set_text("Corrected Bands")
sub3.title.set_text("Iron Content")
sub2.set_xlabel("Pixels")
sub1.set_ylabel("Pixels")
#sub0.imshow(fe_content_1)
sub1.imshow(redness_1)
sub2.imshow(rgb_stack)


###################
#calculate iron content using continuum removal
###################

iron_conent_cont_removal = (-43.3*image_corrected.read(21))+37.9

sub3.imshow(iron_conent_cont_removal)

plt.show()


