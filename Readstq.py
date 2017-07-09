# File: Readstq.py
import datetime


class readSTQFormat:
	
	
	def __init__(self, filename):
			f = open(filename, 'r')
			
			self.table = []
			self.adj_table = []
			self.per_adj_table = []

			day_index = 0
			time_index = 0
			
			day = []
			for next in f:
				if next[0]=='2':
					if next[11]== '2' and next[12]=='2' and next[14]== '0' and next[15]=='0':
						day.append(round((float(next.split(',')[2])+float(next.split(',')[5]))/2, 2))
						time_index += 1
						self.table.append(day)
						day_index += 1
						time_index = 0
						day = []
					else:
						day.append(round((float(next.split(',')[2])+float(next.split(',')[5]))/2, 2))
						time_index += 1

			for i in range(len(self.table)):
				day = []
				for j in range(len(self.table[i])):
					day.append(round(self.table[i][j] - self.table[i][0], 2))
				self.adj_table.append(day)
	
			for i in range(len(self.table)):
				day = []
				for j in range(len(self.table[i])):
					day.append(round(((self.table[i][j] - self.table[i][0])/self.table[i][0])*10000, 0))
				self.per_adj_table.append(day)


