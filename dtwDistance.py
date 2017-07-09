# File: dtwDistance.py
import numpy as np

class DTWDistance:
		
		def __init__(self, w):
			self.w = w
			
			
		def calcDTWDistance(self,s1,s2,w=None):
			DTW={}
    
			if w:
				w = max(w, abs(len(s1)-len(s2)))
    
				for i in range(-1,len(s1)):
					for j in range(-1,len(s2)):
						DTW[(i, j)] = float('inf')
			
			else:
				for i in range(len(s1)):
					DTW[(i, -1)] = float('inf')
				for i in range(len(s2)):
					DTW[(-1, i)] = float('inf')
		
			DTW[(-1, -1)] = 0
	
			for i in range(len(s1)):
				if w:
					for j in range(max(0, i-w), min(len(s2), i+w)):
						dist= (s1[i]-s2[j])**2
						DTW[(i, j)] = dist + min(DTW[(i-1, j)],DTW[(i, j-1)], DTW[(i-1, j-1)])
				else:
					for j in range(len(s2)):
						dist= (s1[i]-s2[j])**2
						DTW[(i, j)] = dist + min(DTW[(i-1, j)],DTW[(i, j-1)], DTW[(i-1, j-1)])
			
			return np.sqrt(DTW[len(s1)-1, len(s2)-1])
			
			
			
		def returnDTWMatrix(self,data):
			DistMatrix = [[0 for i in data] for i in data]
			
			for i, data_i in enumerate(data):
				for j, data_j in enumerate(data):
					DistMatrix[i][j] = self.calcDTWDistance(data_i, data_j, self.w)
			
			return DistMatrix
					
		
		
