#Colm Keyes
#25/11/21
#plotting LAI reflectance for varying soil reflectances.
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
from matplotlib.pyplot import cm

d = []
i=0

####################################
# Matrix manipulations
####################################
csv_location = "C:\\Users\\Colm The Creator\\Documents\\Data\\RemoteSensing_Data\\SAIL simulations LAI-RSL.xls"

lai_soil_matrix = pandas.read_excel(csv_location,sheet_name=[0,1,2])

lai_soil_matrix[1] .iloc[109,:]



lai_sheet1 = lai_soil_matrix[0].drop(columns= "ED=0")
lai_sheet2 = lai_soil_matrix[1].drop(columns= "ED=0")
lai_sheet3 = lai_soil_matrix[2].drop(columns= "ED=0")
columns = lai_sheet1.iloc[4,1:7]
xtiks = [0,0.5,1,2,4,8]

#plt.xticks(,labels=columns)
plt.title("Reflectance for λ=868nm")
plt.xlabel("LAI Value")
plt.ylabel("Reflectance")
plt.scatter(xtiks, lai_soil_matrix[0].iloc[109,1:7], c='r')
plt.plot(xtiks, lai_soil_matrix[0].iloc[109,1:7], c='r',label='RSI=0%')
plt.scatter(xtiks, lai_soil_matrix[1].iloc[109,1:7], c='g')
plt.plot(xtiks, lai_soil_matrix[1].iloc[109,1:7], c='g',label='RSI=10%')
plt.scatter(xtiks, lai_soil_matrix[2].iloc[109,1:7], c='b')
plt.plot(xtiks, lai_soil_matrix[2].iloc[109,1:7], c='b',label='RSI=20%')
plt.legend()#["label1"],["RSI=0%"])#,"RSI=10%","RSI=20%"])
plt.xticks(xtiks)
plt.show()


NIR_reflectance_bands = pd.DataFrame([])
VIS_reflectance_bands = pd.DataFrame([])

for i in [lai_sheet1,lai_sheet2,lai_sheet3]:
#for i in lai_sheet3.columns:

    RSI_values = [0,10,20]
    NIR_wavelength = i.loc[i["Simulations with the PROSPECT-SAILH model"]==868]

    VIS_wavelength = i.loc[i["Simulations with the PROSPECT-SAILH model"]==672]

    NIR_reflectance_bands = NIR_reflectance_bands.append(NIR_wavelength)
    VIS_reflectance_bands = VIS_reflectance_bands.append(VIS_wavelength)

    #dealing with division by zero error
    NIR_reflectance_bands = NIR_reflectance_bands.astype(float)
    VIS_reflectance_bands = VIS_reflectance_bands.astype(float)

DVI = VIS_reflectance_bands.reset_index(drop=True) / NIR_reflectance_bands.reset_index(drop=True)


#plt.xticks(,labels=columns)

color = iter(cm.rainbow(np.linspace(0, 1, 3)))

for i in [0,1,2]:
    c = next(color)
    plt.plot(xtiks[1:], DVI.iloc[i, 2:],label='RSI='+str(i)+"0%",c=c)
    plt.scatter(xtiks[1:], DVI.iloc[i, 2:],c=c)
plt.xticks(xtiks)
plt.legend()
plt.title("DVI Ratio TM3/TM4")
plt.xlabel("LAI Value")
plt.ylabel("Reflectance ratio")
plt.show()

# try:
#     DVI = NIR_reflectance_bands.iloc[:,1:7].astype(float)/VIS_reflectance_bands.iloc[:,1:7].astype(float)
# except ZeroDivisionError:
#     return 0











