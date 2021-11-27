#Colm Keyes
#03/11/21
#Some basic college coding plots - Darkest Pixels

########################
#Imports & Inits
########################
import matplotlib.pyplot as plt

#These values are the minimum value of the statistics read from ERDAS(in order)
dark_pixels_nops86 = [45,14, 10, 5, 1, 1]
dark_pixels_nopa86 = [58, 20, 14, 8, 1, 1]
dark_pixels_nopj86 = [77, 28, 22, 13, 5, 1]
# corresponding Mean wavelength for bands 1-6
band_wavelength = [0.485,0.569,0.660,0.840,1.676,2.223]

########################
#plots
########################
plt.plot(dark_pixels_nops86, band_wavelength,'r', label='Landsat September')
plt.scatter(dark_pixels_nops86,band_wavelength, c='r')
plt.plot(dark_pixels_nopa86,band_wavelength,'g', label='Landsat August')
plt.scatter(dark_pixels_nopa86,band_wavelength, c='g')
plt.plot(dark_pixels_nopj86,band_wavelength,'b', label='Landsat June')
plt.scatter(dark_pixels_nopj86,band_wavelength, c='b')
plt.xlabel('Darkest Pixels')
plt.ylabel('Visual Band(μm)')
plt.legend()
plt.title('Darkest Pixel per Landsat Band')
plt.show()