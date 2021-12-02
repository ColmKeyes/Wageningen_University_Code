#Raimon Bach Pareja
#30/11/21
#Some basic college coding -



import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import seaborn as sns
from sklearn.linear_model import LinearRegression

# GET EXCEL FILE
location = "C:\\Data\\Remote_Sensing\\CourseData\\Remotesensing(1)\\Achterhoek_FieldSpec_2008.xlsx"
matrix = pd.read_excel(location)
first = pd.ExcelFile("C:\\Data\\Remote_Sensing\\CourseData\\Remotesensing(1)\\Achterhoek_FieldSpec_2008.xlsx")
second = pd.read_excel(first, 'Field_sampling')

# matrix.plot(x='Unnamed: 0', y= ['plot', 'Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'])
# matrix.plot(x='Unnamed: 0')
# matrix[['Unnamed: 0', 'plot', 'Unnamed: 2', 'Unnamed: 3']].plot(x='Unnamed: 0')

# CREATE DATA FRAMES
fresh_weight = pd.DataFrame(data=second.iloc[0, 1:])
N_concentration = pd.DataFrame(data=second.iloc[2, 1:])
N_content = pd.DataFrame(data=second.iloc[3, 1:])
ndvi = pd.DataFrame(data=(matrix.iloc[431, :] - matrix.iloc[321, :])
                    / (matrix.iloc[321, :] + matrix.iloc[431, :]))
rep = pd.DataFrame(data=700 + 40 * ((matrix.iloc[321, :] + matrix.iloc[431, :])
                    / 2 - matrix.iloc[351, :]) / (matrix.iloc[391, :] - matrix.iloc[351, :]))
wavelengths = pd.DataFrame(data=matrix.iloc[:, 0][1:])
plot = pd.DataFrame(data=matrix.iloc[:, 1][1:])
plot2 = pd.DataFrame(data=matrix.iloc[:, 2][1:])

# CREATE ARRAYS
list1 = ndvi.values.tolist()
flat_list1 = [j for i in list1 for j in i]
y = np.array(flat_list1[1:])

list5 = rep.values.tolist()
flat_list5 = [j for i in list5 for j in i]
y2 = np.array(flat_list5[1:])

list2 = fresh_weight.values.tolist()
flat_list2 = [j for i in list2 for j in i]
x = np.array(flat_list2)

list3 = N_concentration.values.tolist()
flat_list3 = [j for i in list3 for j in i]
x2 = np.array(flat_list3)

list4 = N_content.values.tolist()
flat_list4 = [j for i in list4 for j in i]
x3 = np.array(flat_list4)

list6 = wavelengths.values.tolist()
flat_list6 = [j for i in list6 for j in i]
x4 = np.array(flat_list6)

list7 = plot.values.tolist()
flat_list7 = [j for i in list7 for j in i]
y4 = np.array(flat_list7)

list8 = plot2.values.tolist()
flat_list8 = [j for i in list8 for j in i]
y5 = np.array(flat_list8)

# DESIGN PLOTS
plt.plot(x, y, 'o')
plt.xlabel('Fresh weight (ton/ha)')
plt.ylabel('NDVI')
plt.title('Regression of NDVI by fresh weight of vegetation')
# plt.text(0.75, 0, 'CC: 0.525', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
plt.grid()
# plt.legend(loc='best', bbox_to_anchor=(0.6, 0.5))
# plt.legend(loc = 'upper left')

# plt.plot(x2, y, 'o')
# plt.xlabel('N concentration (g/kg)')
# plt.ylabel('NDVI')
# plt.title('Regression of NDVI by N concentration in vegetation')
# plt.grid()

# plt.plot(x3, y, 'o')
# plt.xlabel('N content (g/m2)')
# plt.ylabel('NDVI')
# plt.title('Regression of NDVI by N content in vegetation')
# plt.grid()

# plt.plot(x, y2, 'o')
# plt.xlabel('Fresh weight (ton/ha)')
# plt.ylabel('REP')
# plt.title('Regression of REP by fresh weight of vegetation')
# plt.grid()

# plt.plot(x2, y2, 'o')
# plt.xlabel('N concentration (g/kg)')
# plt.ylabel('REP')
# plt.title('Regression of REP by N concentration in vegetation')
# plt.grid()

# plt.plot(x3, y2, 'o')
# plt.xlabel('N content (g/m2)')
# plt.ylabel('REP')
# plt.title('Regression of REP by N content in vegetation')
# plt.grid()

# plt.plot(x3, y4, label='plot1')
# plt.plot(wavelengths, plot2, label='Plot 2')
# plt.xlabel('Wavelength')
# plt.ylabel('Reflectance')
# plt.title('Spectral reflectance of a plot')
# plt.grid()
# plt.legend()
# plt.xticks(np.arange(min(x3), max(x3), 250.0))

# COMPUTE REGRESSION LINE (m = slope, b = intercept)
# m, b = np.polyfit(x3, y2, 1)
# plt.plot(x3, m*x3 + b)

# coef = np.polyfit(x,y,1)
# poly1d_fn = np.poly1d(coef)
# poly1d_fn is now a function which takes in x and returns an estimate for y
# plt.plot(x,y, 'yo', x, poly1d_fn(x), '--k')

slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
print(r_value)
# line = slope * x + intercept
# plt.scatter(x,y)
# plt.plot(x, line, 'r')

# sns.regplot(x, y, ci=None)

plt.show()

# model = LinearRegression().fit(x, y)
# plt.scatter(x, y, color="black")
# plt.plot(x, y, color="blue", linewidth=3)
# plt.show()
