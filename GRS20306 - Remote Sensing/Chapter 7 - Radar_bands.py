#Colm Keyes
#10/11/21
#Some basic college string analysis & retrieving statistics from file
#Analysis of bands within a SAR image, manipulation of band mean values for critical AOI areas of landuse e.g. Potatos 1&2 & Rapeseed.

########################
#Imports & Inits
########################
import re

import pandas
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
d = []
i=0

########################
# Matrix manipulations
#Data was Saved to csv using excel
########################
csv_locations = "C:\\Users\\Colm The Creator\\Documents\\Data\\RemoteSensing_Data\\Land_fields_band_data.csv"
Layers_per_Band = pandas.read_csv(csv_locations, sep=';')
Layers_per_Band = Layers_per_Band.transpose() # flip s.t. every column is a field.
band_freqs = ["3.5cm HH", "3.5cm VV", "3.5cm HV", "23cm HH", "23cm VV", "23cm HV", "60cm HH", "60cm VV",
              "60cm HV"]
Layers_per_Band = Layers_per_Band.rename(index=dict(zip(Layers_per_Band.index, band_freqs)))

########################
#plots
########################
Layers_per_Band.plot(grid=True)
plt.xticks(np.arange(0, 9))
land_types = ["Rapeseed", "Onion", "Peas", "Sugarbeet", "Potato_1", "Potato_2"]
forest_cover = ["Water", "Pine", "Pop(Oxford)", "Pop(Dorskamp)", "Clear cut"]
plt.title("Mean Spectral Signature")
plt.xlabel("Wavelength & Polarisation")
plt.ylabel("Average Signature across Landuse Area")
plt.legend(land_types)
plt.show()