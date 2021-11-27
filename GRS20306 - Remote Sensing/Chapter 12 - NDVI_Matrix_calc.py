#Colm Keyes
#23/11/21
#excel - matrix manipulations of spectral data
#
########################
#Imports & Inits
########################

import re
import pandas
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import numpy
d = []
i=0

####################################
# Matrix manipulations
####################################
csv_location = "C:\\Users\\Colm The Creator\\Documents\\Data\\RemoteSensing_Data\\Achterhoek_FieldSpec_2008.xlsx"

variance_matrix = pandas.read_excel(csv_location)


variance_matrix.plot()
plt.show()

ndvi_df = pd.DataFrame(data=(variance_matrix.iloc[431,:] - variance_matrix.iloc[321,:])/(variance_matrix.iloc[321,:] + variance_matrix.iloc[431,:]))



linear_interp_ndvi_df = pd.DataFrame(data=(700+40*((((variance_matrix.iloc[670,:] + variance_matrix.iloc[780,:])/2) - \
                                                    variance_matrix.iloc[700,:])/(variance_matrix.iloc[740,:]-variance_matrix.iloc[700,:])) ) )

