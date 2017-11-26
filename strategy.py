# File: strategy.py
import os.path
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model, neighbors
from sklearn.naive_bayes import GaussianNB
import numpy.polynomial.polynomial as poly

from ReadGoogle import readGoogleFormat
from Readstq import readSTQFormat

np.set_printoptions(linewidth=120)



entrypoint = 10
outWindowLength = 12
expectedMoveInPricePercent = 80

priceDegree = 2
volumeDegree = 1

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
										   ['Volume Coeff '+str(i) for i in range(volumeDegree+1)]+['classification'])



for index, row in table.iterrows():
	priceCoefs = poly.polyfit([i for i in range(entrypoint+1)], [0.0] + row[XPricecols[:entrypoint]].values.tolist(), priceDegree)
	volumeCoefs = poly.polyfit([i for i in range(entrypoint+1)], [0.0] + [i/10000 for i in row[XVolumecols[:entrypoint]].values], volumeDegree)
	classification = 0
	if max(row[XPricecols[entrypoint:entrypoint+outWindowLength]].values.tolist()) - row[XPricecols[entrypoint-1]] >= expectedMoveInPricePercent:
		classification = 1
	elif row[XPricecols[entrypoint-1]] - min(row[XPricecols[entrypoint:entrypoint+outWindowLength]].values.tolist()) >= expectedMoveInPricePercent:
		classification = -1
	reduced_table.loc[index,:] = priceCoefs.tolist() + volumeCoefs.tolist() + [classification]
	if '2017-05-0' in index:
		print("Day: ", index, [0.0] + row[XPricecols[:entrypoint]].values.tolist(), 'enrty point: ', row[XPricecols[entrypoint-1]])
		print("Coefs: ", priceCoefs.tolist() + volumeCoefs.tolist() + [classification])
		plt.scatter([i+1 for i in range(entrypoint+outWindowLength)], row[XPricecols[:entrypoint+outWindowLength]].values.tolist())
		plt.plot([i+1 for i in range(entrypoint)], [poly.polyval(i+1, priceCoefs) for i in range(entrypoint)],'-')
		plt.show()



print(reduced_table.head())

X = np.array(reduced_table[['Price Coeff '+str(i) for i in range(priceDegree+1)] + ['Volume Coeff '+str(i) for i in range(volumeDegree+1)]])
y = np.array(reduced_table['classification']).astype(int)

numberOfDataPoints, dimensions = X.shape

trainingSize = int(2* numberOfDataPoints/3)

print("Number of Data Points: ", numberOfDataPoints, "Number of Training Examples: ", trainingSize)
X_train = X[:trainingSize]
X_test = X[trainingSize:]
y_train = y[:trainingSize]
y_test = y[trainingSize:]



#scaler = StandardScaler()
#scaler.fit(X_train)



#  CLASSIFICATION OLMUYOR
#  REGRESSION A GERI DON
#



model = GaussianNB() #neighbors.KNeighborsClassifier(n_neighbors=7, weights='distance')
model.fit( #scaler.transform(X_train),
			X_train,
			y_train)

#print(X_test)
#print(model.predict(scaler.transform(X_test)))
testResult = model.predict(X_test)
print(testResult)
print(y_test)
Errcount = sum([1 for i, x in enumerate(testResult) if x != y_test[i] and x != 0])
Rightcount = sum([1 for i, x in enumerate(testResult) if x == y_test[i] != 0 ] )

print('Guessed right: ', Rightcount, ', guessed wrong: ', Errcount)

mycolors =["red", "gold", "limegreen"]
cmap =  colors.ListedColormap(mycolors)
plt.scatter(X_train[:,0], X_train[:,1], c=[i+1 for i in y_train], cmap=cmap)
plt.show()
cmap =  colors.ListedColormap(mycolors)
plt.scatter(X_test[:,0], X_test[:,1], c=[i+1 for i in y_test], cmap=cmap)
plt.show()

'''
X = np.array(table[Xcols])
y = np.array(table[[('Close', i) for (s,i) in table if '16:50:00'<=i<='17:20:00' and s == 'Close']].mean(axis=1))
np.set_printoptions(linewidth=120)
#print(table.iloc[[1]].values)

#print(table.head())



#scaler = StandardScaler()
#scaler.fit(X_train)

#model = LinearSVR()
#model.fit(scaler.transform(X_train), y_train)

print(X_test[:,11])
#print(model.predict(scaler.transform(X_test)))
print(y_test)
#print(model.predict(scaler.transform(X_test)) - X_test[:,11])
print(table[[('Close', i) for (s,i) in table if '16:50:00'<=i<='17:20:00' and s == 'Close']].max(axis=1)[60:].values)
print(table[[('Close', i) for (s,i) in table if '16:50:00'<=i<='17:20:00' and s == 'Close']].min(axis=1)[60:].values.toList)


	#print(model.score(scaler.transform(X_test), y_test))










alldata = red1.per_adj_table + red2.per_adj_table + red3.per_adj_table #+ red4.per_adj_table
print('Total number of days: '+str(len(alldata)))

#print(red1.table)


Coeff = []
Coeff0 = []
Avgs = []
Error = []
xaxis = np.array([i for i in range(len(alldata[0]))])

#print('xaxis: '+str(len(xaxis)))

train_beg = 0 #included
train_end = 15  #not included
test_beg = train_end #included
test_end = len(alldata) #not included

entrypoint = 18 #enter at (entrypoint)th data point. Use range(0:entrypoint) as day opening

opening = [i[0:entrypoint] for i in alldata]
dayendtargetbeg = 50 #included use as index
dayendtargetend = 70 #not included
return_ratio = 0.8 # get in and out of position at expectedreturn*return_ratio

print('Total number of days for training: '+str(train_end-train_beg) + ', Total number of days for test: '+str(test_end - test_beg))
print('Entry Point: '+str(entrypoint) + ', Target Range: '+str(dayendtargetbeg) + '-' + str(dayendtargetend) + ', Return Ratio: ' + str(return_ratio))


for i in range(len(alldata)):
	model = sm.OLS(opening[i], sm.add_constant([j for j in range(len(opening[i]))])).fit()
	Coeff.append(model.params[1])
	Coeff0.append(model.params[0])
	Avgs.append(sum(alldata[i][dayendtargetbeg:dayendtargetend])/(dayendtargetend-dayendtargetbeg))
	error = 0
	for j in range(0,entrypoint):
		error += (Coeff[i]*j - opening[i][j])*(Coeff[i]*j - opening[i][j])
	Error.append(error**(1/2.0))

#Coeff_mean = sum(Coeff)/float(len(Coeff))
#Avgs_mean = sum(Avgs)/float(len(Avgs))

model = sm.OLS(Avgs[train_beg:train_end], sm.add_constant(Coeff[train_beg:train_end])).fit()
#print(model.summary())
w0= model.params[0]
w1= model.params[1]
print('w0: '+str(w0)+', w1: '+str(w1))



plt.scatter(Coeff[train_beg:train_end], Avgs[train_beg:train_end])
for e, x, y in zip(Error[train_beg:train_end], Coeff[train_beg:train_end], Avgs[train_beg:train_end]):
    plt.annotate(round(e, 0), (x, y))
plt.plot([-50, 30], [w0 + w1*(-50), w0 + w1*(30)], 'k-', lw=1)

plt.show()

total_return = 0
for i in range(test_beg,test_end):
	if i%10 == 0:
		model = sm.OLS(Avgs[i-16:i-1], sm.add_constant(Coeff[i-16:i-1])).fit()
		#print(model.summary())
		w0 = model.params[0]
		w1 = model.params[1]
		print('w0: '+str(w0)+', w1: '+str(w1))
		plt.scatter(Coeff[train_beg:train_end], Avgs[train_beg:train_end])
		for e, x, y in zip(Error[train_beg:train_end], Coeff[train_beg:train_end], Avgs[train_beg:train_end]):
			plt.annotate(round(e, 0), (x, y))
		plt.plot([-50, 30], [w0 + w1*(-50), w0 + w1*(30)], 'k-', lw=1)

		plt.show()

	p = alldata[i]
	error = 0
	for j in range(0,entrypoint):
		error += (Coeff[i]*j - opening[i][j])*(Coeff[i]*j - opening[i][j])
	error = error**(1/2.0)
	exp_return = w0 + w1*Coeff[i]
	if exp_return > 0 and p[entrypoint] < exp_return*return_ratio:
		for j in range(entrypoint+1,len(p)):
			if p[j] > exp_return*return_ratio:
				print('won '+str(p[j] - p[entrypoint])+ ' on day ' + str(i) + ', with error: ' + str(round(error,0)))
				print('Got in at: '+str(p[entrypoint])+ ', expected return: ' + str(exp_return) + ', got out at: '+ str(p[j]) + ' on min ' + str(j))
				#plt.plot(xaxis, p)
				#plt.plot([0, entrypoint], [0 , Coeff[i]*entrypoint], 'k-', lw=1)
				#plt.show()
				total_return += p[j] - p[entrypoint]
				break
			elif p[j] < 2*p[entrypoint] - exp_return*return_ratio:
				print('lost '+str(p[j] - p[entrypoint])+ ' on day ' + str(i)+ ', with error: ' + str(round(error,0)))
				print('Got in at: '+str(p[entrypoint])+ ', expected return: ' + str(exp_return) + ', got out at: '+ str(p[j]) + ' on min ' + str(j))
				#plt.plot(xaxis, p)
				#plt.plot([0, entrypoint], [0 , Coeff[i]*entrypoint], 'k-', lw=1)
				#plt.show()
				total_return -= p[entrypoint] - p[j]
				break
	elif exp_return < 0 and p[entrypoint] > exp_return*return_ratio:
		for j in range(entrypoint+1, len(p)):
			if p[j] < exp_return*return_ratio:
				print('won '+str(p[entrypoint] - p[j])+ ' on day ' + str(i)+ ', with error: ' + str(round(error,0)))
				print('Got in at: '+str(p[entrypoint])+ ', expected return: ' + str(exp_return) + ', got out at: '+ str(p[j]) + ' on min ' + str(j))
				#plt.plot(xaxis, p)
				#plt.plot([0, entrypoint], [0, Coeff[i]*entrypoint], 'k-', lw=1)
				#plt.show()
				total_return -= p[j] - p[entrypoint]
				break
			elif p[j] > 2*p[entrypoint] - exp_return*return_ratio:
				print('lost '+str(p[entrypoint] - p[j])+ ' on day ' + str(i)+ ', with error: ' + str(round(error,0)))
				print('Got in at: '+str(p[entrypoint])+ ', expected return: ' + str(exp_return) + ', got out at: '+ str(p[j]) + ' on min ' + str(j))
				#plt.plot(xaxis, p)
				#plt.plot([0, entrypoint], [0 , Coeff[i]*entrypoint], 'k-', lw=1)
				#plt.show()
				total_return += p[entrypoint] - p[j]
				break

print('Total return: '+str(total_return) )

plt.show()

'''
