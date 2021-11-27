#Colm Keyes
#12/11/21
#PCA Analysis using data imported through excel - matrix manipulations
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
import sympy
import numpy
d = []
i=0

####################################
# Matrix manipulations
#Data was Saved to csv using excel
####################################
csv_location = "C:\\Users\\Colm The Creator\\Documents\\Data\\RemoteSensing_Data\\Variance_Calculations.csv"

variance_matrix = pandas.read_csv(csv_location,sep=';')

variance_matrix = variance_matrix.drop(columns=['Unnamed: 4', 'Unnamed: 5', 'Unnamed: 6','Unnamed: 7', 'Unnamed: 8','Unnamed: 9', 'Unnamed: 10', 'Unnamed: 11'], index=[16,17,18])

variance_matrix_split = variance_matrix.pivot(index="Row", columns="col")


variance_matrix["X1-μ1 Std_dev(σ1)"] = variance_matrix.X1 - np.mean(variance_matrix.X1)
variance_matrix["X2-μ2 Std_dev(σ2)"] = variance_matrix.X2 - np.mean(variance_matrix.X2)

####################################
#calculate Variance of each variable from their respective mean.
####################################

#Here we still subtract for each element as the mean of the variance will be zero.

variance_matrix["(X1-μ1)^2_Variance(σ1^2))"] = np.sum((variance_matrix.X1 - np.mean(variance_matrix.X1))**2)/len(variance_matrix)
variance_matrix["(X2-μ2)^2_Variance(σ2^2))"] = np.sum((variance_matrix.X2 - np.mean(variance_matrix.X2))**2)/len(variance_matrix)



####################################
#Calculate Covariance & normalised covariance(correlation coefficient)
#Covariance is the co-variance between two variables, no longer between two elements as the variance may be.
# => we take the sum of the variance averaged over teh variable to get the variance of that variable.
####################################



#so, in our example of ([c11,c21][c12,c22]) the covariance calculated below equals c21 & c12. the squared variance c11
# and c22 will be the variance of each dataset in our new second power **2 plane & axis system.

variance_matrix["(X1-μ1)*(X2-μ2)(Covariance(C))"] = np.sum((variance_matrix.X1 - np.mean(variance_matrix.X1))*((variance_matrix.X2) - np.mean(variance_matrix.X2)))/(len(variance_matrix)-1)
variance_matrix["C1/sqrt(σ1*σ2)(Correl_coeff(Ρ))"] = variance_matrix.iloc[:,8]/np.sqrt(variance_matrix.iloc[:,6]*variance_matrix.iloc[:,7])
varilabels = ["X1","X2"]
#So our covariance matrix C looks like: ([[variance1**2,Covariance],[Covariance,variance2**2]])
covar_matrix = pandas.DataFrame(data=([variance_matrix.iloc[0,6],variance_matrix.iloc[0,8]],[variance_matrix.iloc[0,8],variance_matrix.iloc[0,7]]), columns=varilabels, index=varilabels)
numpy.cross(covar_matrix.iloc[:,0],covar_matrix.iloc[:,1])
# This gives us the right hand side of the equation, I'm not 100% sure how to get the left hand side with the lambda
# To be solved..
# I took a manual approach here to solve for lambda & got lambda1=lambda2 = 156.
eig_values,eig_vectors = np.linalg.eig(covar_matrix)

eigenvalue_matrix = np.diag(eig_values)

a,b = sympy.Matrix(covar_matrix).eigenvects()
normalised_eigvector_a = a[2]
normalised_eigvector_b = b[2]
#These are awkward float within a matrix within a list
eigvector_a = a[2][0]    #*np.mean(variance_matrix.X1)
eigvector_b = b[2][0]    #*np.mean(variance_matrix.X2)

########################
#plots
########################
plt.plot(eig_vectors)
plt.scatter(variance_matrix.iloc[:,4],variance_matrix.iloc[:,5])
plt.plot(eigvector_a)
plt.plot(eigvector_b)
plt.show()

####################################
#eigenvalue matrix: This matrix is a matrix of the variance between two variables less the Covariance that is seen in both variables.
# We remove this information to remove error, and present it as a matrix because our overall goal is to reduce the amount
# of data as little as possible while reducing the number of variables in the system/function(as we could have many variables
# which we do not want to deal with all at the same time)
#(covariance1 - new_variance)*(covariance2 - new_variance) - (covariance12 * covariance21) = 0
#roots = sqrt(b**2-4*a*c)
####################################


####################################
#Calculate eigenvalue matrix(L) from the Covariance matrix(C)
####################################

############################################################################################################
#Section 2
############################################################################################################


txt_location_1 = "C:\\Users\\Colm The Creator\\Documents\\Data\\RemoteSensing_Data\\wg97_model_correl_matrix.mtx"
txt_location_2 = "C:\\Users\\Colm The Creator\\Documents\\Data\\RemoteSensing_Data\\wg95_model_covar_matrix.mtx"


correl = pd.read_csv(txt_location_1,header=None)
covar =  pd.read_csv(txt_location_2,header=None)


upper_triang_correl=np.round(np.triu(correl),2)
lower_triang_correl=np.round(np.tril(correl),2)

upper_triang_covar=np.round(np.triu(covar),2)
lower_triang_covar=np.round(np.tril(covar),2)


lower_triang.tofile('C:\\Users\\Colm The Creator\\Documents\\Data\\RemoteSensing_Data\\lower_triangle.txt', sep = ' ')
np.set_printoptions(threshold=np.inf,linewidth=1000)
pd.set_option('expand_frame_repr', False)