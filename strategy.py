# File: strategy.py
import datetime
import matplotlib.pyplot as plt
import numpy as np
from KMeans import ts_cluster
from dtwDistance import DTWDistance
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import scipy.cluster.hierarchy as hac
from scipy.spatial import distance
from sklearn import linear_model
import statsmodels.api as sm

from ReadGoogle import readGoogleFormat
from Readstq import readSTQFormat







alldata = red1.per_adj_table + red2.per_adj_table + red3.per_adj_table 
print('Total number of days: '+str(len(alldata)))

print(red1.table)


Coeff = []
Coeff0 = []
Avgs = []
Error = []
xaxis = np.array([i for i in range(len(alldata[0]))])

#print('xaxis: '+str(len(xaxis)))

train_beg = 0 #included
train_end = 10  #not included
test_beg = train_end #included
test_end = len(alldata) #not included

entrypoint = 13 #enter at (entrypoint)th data point. Use range(0:entrypoint) as day opening

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

