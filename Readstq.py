# File: Readstq.py
import datetime
import numpy as np




class readSTQFormat:
	
	def __init__(self):
			
			self.table = np.array([])
			self.adj_table = []
			self.per_adj_table = []

			allFiles = [r'data\5min-03may2017-19may2017\us\nasdaq stocks\2\tsla.us.txt', 
						#r'data\5min-22may2017-07june2017\us\nasdaq stocks\2\tsla.us.txt',
						#r'data\5min-08june2017-23june2017\us\nasdaq stocks\2\tsla.us.txt',
						r'data\5min-26june2017-12july2017\us\nasdaq stocks\2\tsla.us.txt'
						]
			
			for filename in allFiles:
				self.readSingleFile(filename)

	def readSingleFile(self, filename):
			
			values = np.loadtxt(filename, delimiter=',', usecols=[2,3,4,5,6], skiprows=1)
			dateTimes = np.loadtxt(filename, delimiter=',', usecols=[0,1], skiprows=1, dtype = np.str)
			print(values)
			print(dateTimes)
			nRow, nCol = values.shape
			print(nRow)
			print(nCol)
			if nRow%78 == 0:
				print("yes")
				reshape(values[:,0], (78, nRow/78))
				print(values)
			else:
				print("no")
				
			day_index = 0
			time_index = 0
			
			day = []
			newd = 0
			for i in range(nRow):
				if dateTimes[0][1]=='22:00:00':
						newd = 1
				else

	

red1 = readSTQFormat()
