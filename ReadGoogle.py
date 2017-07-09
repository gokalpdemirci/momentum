# File: ReadGoogle.py
import datetime


class readGoogleFormat:
	
	
	def __init__(self, filename):
			f = open(filename, 'r')
			'''next = f.readline()
			print(next.strip())
			next = f.readline()
			print(next.strip())
			next = f.readline()
			print(next.strip())
			next = f.readline()
			print(next.strip())
			next = f.readline()
			print(next.strip())
			next = f.readline()
			print(next.strip())'''
			
			self.table = []
			self.adj_table = []
			self.per_adj_table = []

			time_offset = 0
			before = 0
			time_now = 0
			day_index = -1
			time_index = 0
			first_day = 1
			
			day = []
			for next in f:
				if next[0]=='T':
					time_now = time_now - time_offset*60
					time_offset = int(next[16:20])
					time_now = time_now + time_offset*60
				elif next[0]=='a':
					time_now = int(next[1:11]) + time_offset*60
					before = 0
					utc_time = datetime.datetime.utcfromtimestamp(time_now)#.strftime('%Y-%m-%d %H:%M:%S')
					if utc_time.hour == 9 and utc_time.minute == 30:
						day_index += 1
						time_index = 0
						if day_index > 0:
							self.table.append(day)
						day = []
					elif time_index != 1 and utc_time.hour == 9 and utc_time.minute == 35:
						day_index += 1
						time_index = 1
						if day_index > 0:
							self.table.append(day)
						day = []
						day.append(round((float(next.split(',')[1])+float(next.split(',')[4]))/2, 2))
						
					day.append(round((float(next.split(',')[1])+float(next.split(',')[4]))/2, 2))
					time_index += 1
					#print(utc_time, day)
				elif next[0]<= '9' and next[0]>='0':
					time_now += 5*60*(int(next.partition(',')[0]) - before)
					before = int(next.partition(',')[0])
					utc_time = datetime.datetime.utcfromtimestamp(time_now)
					if utc_time.hour == 9 and utc_time.minute == 30:
						day_index += 1
						time_index = 0
						if day_index > 0:
							self.table.append(day)
						day = []
					elif time_index != 1 and utc_time.hour == 9 and utc_time.minute == 35:
						day_index += 1
						time_index = 1
						if day_index > 0:
							self.table.append(day)
						day = []
						day.append(round((float(next.split(',')[1])+float(next.split(',')[4]))/2, 2))
					
					day.append(round((float(next.split(',')[1])+float(next.split(',')[4]))/2, 2))
					time_index += 1
					#print(utc_time, day)

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


