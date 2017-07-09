# File: testClustering.py
import datetime
import matplotlib.pyplot as plt
import numpy as np
from KMeans import ts_cluster








c = ts_cluster(4)
c.k_means_clust(red.per_adj_table,10,50,4)
assignments = c.get_assignments()



x = np.array([i for i in range(79)])

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
for i in assignments[0]:
	ax1.plot(x, red.per_adj_table[i])

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
for i in assignments[0]:
	ax1.plot(x, red.per_adj_table[i])

fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
for i in assignments[1]:
	ax2.plot(x, red.per_adj_table[i])

fig3 = plt.figure()
ax3 = fig3.add_subplot(111)
for i in assignments[2]:
	ax3.plot(x, red.per_adj_table[i])

fig4 = plt.figure()
ax4 = fig4.add_subplot(111)
for i in assignments[3]:
	ax4.plot(x, red.per_adj_table[i])

plt.show()


