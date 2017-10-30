# File: Readstq.py
import datetime
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.svm import LinearSVR





class readSTQFormat:

	def __init__(self):

			totalList = []
			dates = []
			self.table = np.array([])
			self.adj_table = []
			self.per_adj_table = []

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

						]




			values = [pd.read_csv(filename, usecols=[0,1,2,5,6]) for filename in allFiles]
			pivotted_values = [df.pivot(index='Date', columns='Time') for df in values]
			table = pd.concat(pivotted_values)


			for (s,i) in table:
				if s == 'Open' and i != '15:35:00':
					del table[('Open', i)]

			if table[('Open', '15:35:00')].isnull().values.any():
				print('Opening values have non!')

			for (s,i) in table:
				if s == 'Close':
					table[('Close', i)] = 10000*(table[('Close', i)] - table[('Open', '15:35:00')])/table[('Open', '15:35:00')]
			table.dropna(inplace = True)
			#table.fillna(-99999, inplace=True)
			Xcols = [('Close', i) for (s,i) in table if '15:35:00'<=i<='16:30:00' and s == 'Close']
			Xcols += [('Volume', i) for (s,i) in table if '15:35:00'<=i<='16:30:00' and s == 'Volume']

			#print(Xcols)
			#print(table[Xcols].head())

			X = np.array(table[Xcols])
			y = np.array(table[[('Close', i) for (s,i) in table if '16:50:00'<=i<='17:20:00' and s == 'Close']].mean(axis=1))

			np.set_printoptions(linewidth=120)
			#print(table.iloc[[1]].values)
			#print(X[1])
			#print(y[1])
			print(X.shape)
			#print(table.head())

			X_train = X[:60]
			X_test = X[60:]
			y_train = y[:60]
			y_test = y[60:]

			scaler = StandardScaler()
			scaler.fit(X_train)

			model = LinearSVR()
			model.fit(scaler.transform(X_train), y_train)

			print(X_test[:,11])
			print(model.predict(scaler.transform(X_test)))
			print(y_test)
			print(model.predict(scaler.transform(X_test)) - X_test[:,11])
			print(table[[('Close', i) for (s,i) in table if '16:50:00'<=i<='17:20:00' and s == 'Close']].max(axis=1)[60:].values)
			print(table[[('Close', i) for (s,i) in table if '16:50:00'<=i<='17:20:00' and s == 'Close']].min(axis=1)[60:].values)


			print(model.score(scaler.transform(X_test), y_test))


			"""" THIS WORKS GREAT UNTIL HERE
					HANDLE NANs IN THE TABLE AND JOIN ALL TABLES
					NOW EACH ROW IN THE TABLE IS 1 DAY AND EACH COLUMN IS A PARTICULAR TIME OF THE DAY"""


	"""			dates.extend(d)
				totalList.extend(l)


			if len(totalList)%78 == 0:
				self.table = np.array(totalList)
				self.table.shape =  (int(len(totalList)/78), 78)
				print('\n Files read succesfully! Number of days: ' + str(len(self.table)))
			else:
				print('Error: Total List length is not a multiple of 78.')
	"""




	"""
	def readSingleFile(self, filename):

			values = np.loadtxt(filename, delimiter=',', usecols=[2], skiprows=1)
			dateTimes = np.loadtxt(filename, delimiter=',', usecols=[0,1], skiprows=1, dtype = np.str)
			dates = np.unique(dateTimes[:,0])
			valuesList = values.tolist()



			values = pd.read_csv(filename, usecols=[0,1,2])
			values = values.pivot(index='Date', columns='Time', values='Open')


			print(values.head())





				if len(values)%78 == 0:
				print("File seems to be clean: " + filename)
			else: # clean data
				print("File has missing data: " + filename)
				offset = 0 #offset between indexing valuesList and values as we go on updating one and keeping the other smaller
				prev = 0
				for i in range(1, len(values)):
					differ = datetime.datetime.strptime(dateTimes[i][1], '%H:%M:%S') - datetime.datetime.strptime(dateTimes[prev][1], '%H:%M:%S')
					if  differ.seconds!=300 and differ.seconds!=63300:
						if dateTimes[i][0] == dateTimes[prev][0]:
							numberofmissing = int((differ.seconds/300) - 1)
							print("Missing values " + str(numberofmissing) + " values between " + dateTimes[prev][0] +' ' + dateTimes[prev][1] + " and " + dateTimes[i][0] + ' ' + dateTimes[i][1])
							end1 = values[prev]
							end2 = values[i]
							for j in range(numberofmissing):
								valuesList.insert(offset+j+prev+1, end1 + (end2-end1)/(numberofmissing +1))
							offset += numberofmissing
							print("Corrected " + str(numberofmissing) + " missing values between " + dateTimes[prev][0] +' ' + dateTimes[prev][1] + " and " + dateTimes[i][0] + ' ' + dateTimes[i][1])
						elif dateTimes[i][1] == '15:35:00' and  dateTimes[prev][1] != '22:00:00':
							differ = datetime.datetime.strptime('22:05:00', '%H:%M:%S') - datetime.datetime.strptime(dateTimes[prev][1], '%H:%M:%S')
							numberofmissing = int((differ.seconds/300) - 1)
							print("Missing values " + str(numberofmissing) + " values between " + dateTimes[prev][0] +' ' + dateTimes[prev][1] + " and " + dateTimes[i][0] + ' ' + dateTimes[i][1])
							for j in range(numberofmissing):
								valuesList.insert(offset+j+prev+1, values[prev])
							offset += numberofmissing
							print("Corrected " + str(numberofmissing) + " missing values between " + dateTimes[prev][0] +' ' + dateTimes[prev][1] + " and " + dateTimes[i][0] + ' ' + dateTimes[i][1])
						elif dateTimes[i][1] != '15:35:00' and  dateTimes[prev][1] == '22:00:00':
							differ = datetime.datetime.strptime(dateTimes[i][1], '%H:%M:%S') - datetime.datetime.strptime('15:35:00', '%H:%M:%S')
							numberofmissing = int((differ.seconds/300) - 1)
							print("Missing values " + str(numberofmissing) + " values between " + dateTimes[prev][0] +' ' + dateTimes[prev][1] + " and " + dateTimes[i][0] + ' ' + dateTimes[i][1])
							for j in range(numberofmissing):
								valuesList.insert(offset+i-1-j, values[i])
							offset += numberofmissing
							print("Corrected " + str(numberofmissing) + " missing values between " + dateTimes[prev][0] +' ' + dateTimes[prev][1] + " and " + dateTimes[i][0] + ' ' + dateTimes[i][1])
						elif dateTimes[i][1] != '15:35:00' and  dateTimes[prev][1] != '22:00:00':
							differ = datetime.datetime.strptime('22:05:00', '%H:%M:%S') - datetime.datetime.strptime(dateTimes[prev][1], '%H:%M:%S')
							numberofmissing = int((differ.seconds/300) - 1)
							print("Missing values " + str(numberofmissing) + " values between " + dateTimes[prev][0] +' ' + dateTimes[prev][1] + " and " + dateTimes[i][0] + ' ' + dateTimes[i][1])
							for j in range(numberofmissing):
								valuesList.insert(offset+j+prev+1, values[prev])
							offset += numberofmissing
							print("Corrected " + str(numberofmissing) + " missing values between " + dateTimes[prev][0] +' ' + dateTimes[prev][1] + " and " + dateTimes[i][0] + ' ' + dateTimes[i][1])

							differ = datetime.datetime.strptime(dateTimes[i][1], '%H:%M:%S') - datetime.datetime.strptime('15:35:00', '%H:%M:%S')
							numberofmissing = int((differ.seconds/300) - 1)
							print("Missing values " + str(numberofmissing) + " values between " + dateTimes[prev][0] +' ' + dateTimes[prev][1] + " and " + dateTimes[i][0] + ' ' + dateTimes[i][1])
							for j in range(numberofmissing):
								valuesList.insert(offset+i-1-j, values[i])
							offset += numberofmissing
							print("Corrected " + str(numberofmissing) + " missing values between " + dateTimes[prev][0] +' ' + dateTimes[prev][1] + " and " + dateTimes[i][0] + ' ' + dateTimes[i][1])

						#else:
							#handle missing days
					prev = i

			if len(valuesList)%78 != 0:
				print('List length is not a multiple of 78: ' + str(len(valuesList)) + ' in ' + filename)

			return dates, valuesList
	"""





if __name__ == '__main__':
	red1 = readSTQFormat()
