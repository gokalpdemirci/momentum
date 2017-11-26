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

	def __init__(self, allFiles):

			totalList = []
			dates = []
			self.adj_table = []
			self.per_adj_table = []


			values = [pd.read_csv(filename, usecols=[0,1,2,5,6]) for filename in allFiles]
			pivotted_values = [df.pivot(index='Date', columns='Time') for df in values]
			table = pd.concat(pivotted_values)


			for (s,i) in table:
				if s == 'Open' and i != '15:35:00':
					del table[('Open', i)]

			#print(table)

			if table[('Open', '15:35:00')].isnull().values.any():
				print('Readstq Warning: Opening values have non!')

			table.dropna(inplace = True)
			#table.fillna(-99999, inplace=True)

			for (s,i) in table:
				if s == 'Close':
					table[('Close', i)] = 10000*(table[('Close', i)] - table[('Open', '15:35:00')])/table[('Open', '15:35:00')]

			self.table = table




if __name__ == '__main__':
	red1 = readSTQFormat(allFiles = [r'data\5min-03may2017-19may2017\us\nasdaq stocks\2\tsla.us.txt',
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

						])
