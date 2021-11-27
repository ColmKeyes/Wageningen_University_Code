#Colm Keyes
#10/11/21
#Some basic college string analysis & retrieving statistics from file
# regex search followed by plotting

########################
#Imports & Inits
########################
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
d = []
i=0

########################
#txt to dataframe manipulations
########################
f = open("C:\\Users\\Colm The Creator\\Documents\\Data\\RemoteSensing_Data\\ahs_po1_image_statistics.txt")
g = open("C:\\Users\\Colm The Creator\\Documents\\Data\\RemoteSensing_Data\\ahs_po1_image_noise_statistics.txt")
stats_1 = f.readlines()
stats_2 = g.readlines()

dev_dict_1 = pd.DataFrame(stats_1)
dev_dict_2 = pd.DataFrame(stats_2)

std_dev = []

################################################
#loop to find the Std.Dev values from .txt file f
################################################
for i in range(len(dev_dict_1)):
    stat = dev_dict_1[0][i]
    find = []
    find = re.search('Deviation', stat)
    if find:
        std_dev.append(stat.split()[3])

std_dev_noise = []

################################################
#loop to find the Std.Dev values from .txt file g
################################################
for i in range(len(dev_dict_2)):
    stat = dev_dict_2[0][i]
    find = []
    find = re.search('Deviation', stat)
    if find:
        std_dev_noise.append(stat.split()[3])

std_dev = list(map(float, std_dev))
std_dev_noise = list(map(float, std_dev_noise))

bands = np.arange(0,len(std_dev))

snr = [a_i / b_i for a_i, b_i in zip(std_dev, std_dev_noise)]

#################
#Plots of Signal, Noise & SNR
#################

plt.plot(bands, std_dev, label='Signal')
plt.plot(bands, std_dev_noise,'r', label='Noise')
plt.ylabel('Std Deviation')
plt.xlabel('Band No.')
plt.legend()
plt.title('Std Deviation of whole image bands')
plt.show()



plt.plot(bands, snr, 'b',label='SNR')
plt.ylabel('SNR')
plt.xlabel('Band No.')
plt.legend()
plt.title('SNR per Image Band')
plt.show()