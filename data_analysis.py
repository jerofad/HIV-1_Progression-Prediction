# -*- coding: utf-8 -*-

##### Library Importation ####

import seaborn as sns
import numpy as np
import scipy
import pandas as pd
from matplotlib import pyplot as plt

#get_ipython().run_line_magic('matplotlib', 'inline')
#########################################################################

##### load training dataset ######
data =  pd.read_csv('training_data.csv', delimiter=',')
print(data[['Resp','PR Seq','RT Seq','VL-t0', 'CD4-t0']].head())


# general information about the data
data.info()


####### Number of missing data ########
data.isnull().sum().sum()


####### Number of missing data ########
print('Missing data:')
print(data[['Resp','PR Seq','RT Seq','VL-t0', 'CD4-t0']].isnull().sum())


data_X = data[['VL-t0', 'CD4-t0']]


# descriptive statistic
data[['Resp','PR Seq','RT Seq','VL-t0', 'CD4-t0']].describe()


################ correlation matrix of data_X ############
#data[['VL-t0', 'CD4-t0']].corr()
print('Correlation Matrix \n')
data_X.corr()


# general data correlation heatmap
sns.heatmap(data[['Resp','VL-t0', 'CD4-t0']].corr(),cmap='YlGnBu')
plt.title('Correlation heatmap')
plt.savefig('corr_matrix_plot')


##############  Histogram of improving '1' of responds after 16 weeks of therapy #################
#data['Resp'].hist()
data["Resp"].value_counts().plot.pie()
#plt.ylabel("nb patient")
#plt.xlabel("improve 1 or not 0")
plt.title('Pie Plot of Class 0 and 1: 0 = no improvement')
plt.savefig('imbalanced_plot')



N,D = data.shape
first_pct = np.sum(data.iloc[:,1] == 0)/N
second_pct = np.sum(data.iloc[:,1] == 1)/N

print('First Class precentage: %{:2f}'.format(100*first_pct))
print('Second Class precentage: %{:2f}'.format(100*second_pct))


# group the data by the target variable 
data_grp= data[['Resp','VL-t0', 'CD4-t0']].groupby('Resp')


# general descriptive per-class
data_grp.describe()


##################################### Class 0: no improvement  ###################################################

class_0 = data[data['Resp']==0]

class_0_X = class_0[['VL-t0','CD4-t0']]

# print('Class 0')
# class_0_X.describe()

#### Correlation matrix of class 0
print('Class 0')
class_0_X.corr()

######## CD4+ count histogram: class 0
class_0_X['CD4-t0'].hist()
plt.ylabel('Effective')
plt.xlabel('CD4+ count')
plt.title('Class 0: no improvement')
plt.savefig('Class0_origin_data_CD4')

###### VL-t0 histogram
class_0_X['VL-t0'].hist()
plt.ylabel('Effective')
plt.xlabel('viral load')
plt.title('Class 0: no improvement')
plt.savefig('Class0_origin_data_VL')


############################ Class 1: Improvement ##################################################

class_1 = data[data['Resp']==1]

class_1_X = class_1[['VL-t0','CD4-t0']]

# print('Class 1')
# class_1_X.describe()


######## Correlation matrix of class 1
print('Class 1')
class_1_X.corr()


###### CD4+ count histogram

class_1_X['CD4-t0'].hist()
plt.ylabel('Effective')
plt.xlabel('CD4+ count')
plt.title('Class 1: improvement')
plt.savefig('Class0_origin_data_CD4')


###### VL-t0 histogram

class_1_X['VL-t0'].hist()
plt.ylabel('Effective')
plt.xlabel('viral load')
plt.title('Class 1: improvement')
plt.savefig('Class1_origin_data_VL')


########################################### Plotting  #################################################


##############  plotting of viral load "VL-t0" (all ) #################
data_grp['VL-t0'].plot()
plt.xlabel("nb patient")
plt.ylabel("viral load")

######  save figure, format '.png'
plt.savefig('viral_load_plot.png')


##############  plotting of absolute CD4 count "CD4-t0" #################
data_grp['CD4-t0'].plot()

plt.axhline(500, 0, 1002, color='r')
plt.xlabel("nb patient")
plt.ylabel("CD4+ count")


############ pairplot ##########

sns.pairplot(data, hue='Resp')


