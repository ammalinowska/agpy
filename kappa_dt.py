from __future__ import division
import matplotlib as mpl
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import math
import pandas as pd

sns.set_palette('rocket')

df=pd.read_csv(r'Single_Export_csv/9.csv',usecols= ['x', 'y', 'sx', 'sy', 'curvature', 'mean(curvature)', 'contour_num', 'frame', 'area', 'length'])
	

	
unique_val =  pd.unique(df.frame)
df2 = []
lengths = []
for i in unique_val:
 	df_i = df.loc[lambda df:df.frame == i]
 	df_i.reset_index()
 	# print(df_i)
 	space_scale = 87*10**(-3)
 	a=df_i.curvature
 	ylist = a.values.tolist()#)  select y values (in this case curvature)
 	df3 = pd.DataFrame(ylist, columns=[i])
 	lengths.append(len(df3))

 	df2.append(df3)
 	df_len = pd.DataFrame(lengths)
 	ylist = [x * 1/(space_scale) for x in ylist]
 	# plt.plot(ylist)
 	# plt.legend()
 	plt.xlabel('contour (a.u.)')
 	plt.ylabel('curvature (1/um)')
 	all_curvatures = pd.concat(df2, axis=1)

# print(lengths)
# print(df_len)
max_len = max(lengths)
print(max(lengths))
# print(all_curvatures)
# all_curvatures.dropna()
print(all_curvatures)

label=[]
plt.figure()

#rescale curves so they all have equal amount of datapoints
for j in all_curvatures.dropna():
	
	label.append(j)
	
	ycolumn = all_curvatures[j].dropna()
	x = range(0, len(ycolumn))
	xlong = np.arange(0,max_len)
	xshort = np.arange(0,len(ycolumn))
	scale = (max_len/len(ycolumn))
	xscaled = [(x * scale) for x in xshort]
	# print(len(xscaled))
	# plt.figure()
	# plt.plot(xlong, all_curvatures[0])
	# plt.plot(xscaled, ycolumn)
	# plt.title('Rescaled curvatures')
	# plt.xlabel('Contour (a.u.)')
	# plt.ylabel('Curvature (1/um)')
	# # print(label)	
	# plt.legend(label)

	# To be able to track curvature changes in time, the same points of the crystal should be compared;
	# to do that we shift the curves in respect to the longest and calculate the differences. 
	# The offset with the lowest difference will be picked as the most accurate. 
	differences_j = []
	offset_j = []
	longest = all_curvatures[0] #temporary fix

	for shift in np.arange(0,len(ycolumn),1):
		y_new = np.pad(ycolumn, (int(shift),0), 'wrap')[:len(ycolumn)]
		diff = [abs(longest[int(xscaled[i])] - y_new[i]) for i in range(0,len(ycolumn))]
		differences_j.append(sum(diff))
		offset_j.append(shift)

# Define curvature curve with lowest difference
	final_offset = offset_j[differences_j.index(min(differences_j))]
	y_best = np.pad(ycolumn, (int(final_offset),0), 'wrap')[:len(ycolumn)]
	print(f'The offset that gives the minimal difference for frame {j} is {final_offset}')
    

    
#   
# Plot what the result looks like
	# plt.figure(figsize=(10,5))
	# plt.subplot(1,2,1)
	# plt.title('Difference between curves')
	# plt.ylabel('Difference (1/um)')
	# plt.xlabel('Offset (a.u.)')
	# plt.plot(offset_j,differences_j)
	# plt.subplot(1,2,2)
	plt.xlabel('Contour (a.u.)')
	plt.ylabel('Curvature (1/um)')
	plt.title('Minimal difference result')
	plt.plot(longest)
	plt.legend(label)
	plt.plot(xscaled, y_best)
	plt.text((max_len/2), max(longest), 'Offset=%d'%(final_offset), color='red')
plt.show()
