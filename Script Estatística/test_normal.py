import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import scikit_posthocs as sp

from scipy import stats
from sklearn import preprocessing



print ("RAW DATA")
df = pd.read_excel (r'.\RESULTADOS - Copia.xlsx', sheet_name='Raw')
print (df.columns.values)

print ("\nNORMALIZED DATA - DIVIDE EACH LINE BY ITS MAX VALUE")
#normalized = np.transpose(preprocessing.normalize(df, norm='max', axis=1))
normalized = np.transpose(df.to_numpy())
print(normalized)

bestVal = np.min(np.sum(normalized, axis=1))
bestValIndex = np.argmin(np.sum(normalized, axis=1))

x, y = np.shape(normalized)

print ("\nNORMAL DISTRIBUTION TEST FOR EACH ALGORITHM - SHAPIRO-WILK")
isNormal = True
alpha = 0.05

for i in range(0, x):
	
	stat, p = stats.shapiro(normalized[i,:])
	print(df.columns.values[i] + " p = " + str(p) + " w = " + str(stat))

	if p > alpha:
		print('\tSample looks Gaussian (fail to reject H0)\n')
	else:
		print('\tSample does not look Gaussian (reject H0)\n')
		isNormal = False


#print(*normalized)
isEqual = True

if isNormal:
	
	print ("\nTEST POPULATION MEDIAN - ANOVA ONE-WAY")
	#for i in range(0, y):
		#if i != bestValIndex:
	sts, pval = stats.f_oneway(*normalized)
	print("pval = " + str(pval))

	if pval > alpha:
		print('\tSets are related (fail to reject H0)\n')
	else:
		print('\tSets differ (reject H0)\n')
		isEqual = False
				
	if not isEqual:
		for i in range(0, x):
			if i != bestValIndex:
				sts, pval = stats.ttest_ind(normalized[:,bestValIndex], normalized[:, i])
				if pval > alpha:
					print('\tSets '+ str(bestValIndex) + ', ' + str(i) + ' are related (fail to reject H0)\n')
				else:
					print('\tSets '+ str(bestValIndex) + ', ' + str(i) + ' differ (reject H0)\n')
	#stats.ttest_ind
else:
	print ("\nTEST POPULATION MEDIAN - KRUSKAL-WALLIS")
	#for i in range(0, y):
		#if i != bestValIndex:
	sts, pval = stats.kruskal(*normalized)
	print("pval = " + str(pval))

	if pval > alpha:
		print('\tSets are related (fail to reject H0)\n')
	else:
		print('\tSets differ (reject H0)\n')
		isEqual = False

	if not isEqual:
		print ("\nTEST POPULATION - WILCOXON")
		for i in range(0, x):
			if i != bestValIndex:
				sts, pval = stats.wilcoxon(normalized[bestValIndex,:], normalized[i, :])
				if pval > alpha:
					print('\tSets '+ str(bestValIndex) + ', ' + str(i) + ' are related (fail to reject H0)\n')
				else:
					print('\tSets '+ str(bestValIndex) + ', ' + str(i) + ' differ (reject H0)\n')
		
		print ("\nTEST POPULATION - DUNN")
		pvals = sp.posthoc_dunn(normalized)
		print(pvals)
		
	



