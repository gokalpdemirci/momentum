# File: strategy.py
import os.path
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model, neighbors
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVR

import numpy.polynomial.polynomial as poly

from ReadGoogle import readGoogleFormat
from Readstq import readSTQFormat

np.set_printoptions(linewidth=120)



entrypoint = 8
outWindowLength = 6
expectedMoveInPricePercent = 80

priceDegree = 2
volumeDegree = 2

allFiles = [r'data\5min-03may2017-19may2017\us\nasdaq stocks\2\tsla.us.txt',
			r'data\5min-22may2017-07june2017\us\nasdaq stocks\2\tsla.us.txt',
			r'data\5min-08june2017-23june2017\us\nasdaq stocks\2\tsla.us.txt',
			r'data\5min-26june2017-13july2017\us\nasdaq stocks\2\tsla.us.txt',
			r'data\5min-13july2017-28july2017\us\nasdaq stocks\2\tsla.us.txt',
			r'data\5min-31july2017-15august2017\us\nasdaq stocks\2\tsla.us.txt',
			r'data\5min-17august2017-01september2017\us\nasdaq stocks\2\tsla.us.txt',
			r'data\5min-05september2017-19september2017\us\nasdaq stocks\2\tsla.us.txt',
			r'data\5min-21september2017-06october2017\us\nasdaq stocks\2\tsla.us.txt',
			r'data\5min-09october2017-24october2017\us\nasdaq stocks\2\tsla.us.txt',
			#starts with 14:35:00 instead of 15:35:00	r'data\5min-25october2017-09november2017\us\nasdaq stocks\2\tsla.us.txt',
			]


if os.path.isfile('tempdata.csv'):
	table = pd.read_csv('tempdata.csv', skipinitialspace=True, index_col=[0], header=[0, 1])
else:
	read = readSTQFormat(allFiles)
	read.table.to_csv('tempdata.csv')
	table = read.table

print(table.head())

XPricecols = [('Close', i) for (s,i) in table if '15:35:00'<=i<='20:00:00' and s == 'Close']
XVolumecols = [('Volume', i) for (s,i) in table if '15:35:00'<=i<='20:00:00' and s == 'Volume']

print("XPriceColumns: ", XPricecols)
print(table[XPricecols].head())

reduced_table = pd.DataFrame(index= table.index.values, columns = ['Price Coeff '+str(i) for i in range(priceDegree+1)]+
										   ['Volume Coeff '+str(i) for i in range(volumeDegree+1)]+['classification', 'Regression Target', 'Regression Diff to entrypoint'])



for index, row in table.iterrows():
	priceCoefs = poly.polyfit([i for i in range(entrypoint+1)], [0.0] + row[XPricecols[:entrypoint]].values.tolist(), priceDegree)
	volumeCoefs = poly.polyfit([i for i in range(entrypoint+1)], [0.0] + [i/10000 for i in row[XVolumecols[:entrypoint]].values], volumeDegree)
	classification = 0
	if max(row[XPricecols[entrypoint:entrypoint+outWindowLength]].values.tolist()) - row[XPricecols[entrypoint-1]] >= expectedMoveInPricePercent:
		classification = 1
	elif row[XPricecols[entrypoint-1]] - min(row[XPricecols[entrypoint:entrypoint+outWindowLength]].values.tolist()) >= expectedMoveInPricePercent:
		classification = -1
	regression_target = sum(row[XPricecols[entrypoint+outWindowLength-3:entrypoint+outWindowLength]].values.tolist())/3.0
	regression_diff = regression_target - row[XPricecols[entrypoint-1]]
	reduced_table.loc[index,:] = priceCoefs.tolist() + volumeCoefs.tolist() + [classification, regression_target, regression_diff]
	if '2017-05-0' in index and False:
		print("Day: ", index, [0.0] + row[XPricecols[:entrypoint]].values.tolist(), 'enrty point: ', row[XPricecols[entrypoint-1]])
		print(row[XPricecols[:entrypoint+outWindowLength]].values.tolist())
		print("Coefs: ", priceCoefs.tolist() + volumeCoefs.tolist() + [classification, regression_target, regression_diff])
		plt.scatter([i+1 for i in range(entrypoint+outWindowLength)], row[XPricecols[:entrypoint+outWindowLength]].values.tolist())
		plt.plot([i+1 for i in range(entrypoint)], [poly.polyval(i+1, priceCoefs) for i in range(entrypoint)],'-')
		plt.show()



print(reduced_table.head())


X = np.array(reduced_table[['Price Coeff '+str(i) for i in range(0, priceDegree+1)] + ['Volume Coeff '+str(i) for i in range(0, volumeDegree+1)]])
y = np.array(reduced_table['Regression Diff to entrypoint'])

print(X)
print(y)

numberOfDataPoints, dimensions = X.shape

trainingSize = int(5* numberOfDataPoints/6)

print("Number of Data Points: ", numberOfDataPoints, "Number of Training Examples: ", trainingSize)
X_train = X[:trainingSize]
X_test = X[trainingSize:]
y_train = y[:trainingSize]
y_test = y[trainingSize:]



#scaler = StandardScaler()
#scaler.fit(X_train)


#  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#  EVENT ACISINDAN BAK OLAYA
#  GUN ACILISINDA HIZLI DIK BIR SEKILDE BASLAMIYORSA ZATEN BIR SEY OLMUYOR O GUN
#  BU NOISE YAPIYOR. COEFF I COK BUYUK VEYA COK KUCUK OLMAYAN ROW LARI AT
#  SONRA REGRESSION YAP!



model = linear_model.LinearRegression(normalize=True) #SVR(kernel='rbf') #GaussianNB() #neighbors.KNeighborsClassifier(n_neighbors=7, weights='distance')
model.fit( #scaler.transform(X_train),
			X_train,
			y_train)

#print(X_test)
#testResult = model.predict(scaler.transform(X_test))
testResult = model.predict(X_test)
print(testResult)
print(y_test)
Errcount = sum([1 for i, x in enumerate(testResult) if (x>50 and y_test[i]<30) or (x<-50 and y_test[i]>-30)])
Rightcount = sum([1 for i, x in enumerate(testResult) if (x>50 and y_test[i]>30) or (x<-50 and y_test[i]<-30)])

print('Guessed right: ', Rightcount, ', guessed wrong: ', Errcount)

mycolors =["red", "gold", "limegreen"]
cmap =  colors.ListedColormap(mycolors)
#plt.scatter(X_train[:,0], X_train[:,1], c=[int(i>80)*0 + int(-80<=i<=80)*1 + int(i<-80)*2 for i in y_train], cmap=cmap)
plt.scatter(X_train[:,0], y_train)
plt.show()
cmap =  colors.ListedColormap(mycolors)
#plt.scatter(X_test[:,1], X_test[:,1], c=[int(i>80)*0 + int(-80<=i<=80)*1 + int(i<-80)*2 for i in y_test], cmap=cmap)
plt.scatter(X_train[:,1], y_train)
plt.show()
plt.scatter(X_train[:,2], y_train)
plt.show()
plt.scatter(X_train[:,3], y_train)
plt.show()
plt.scatter(X_train[:,4], y_train)
plt.show()
plt.scatter(X_train[:,5], y_train)
plt.show()
