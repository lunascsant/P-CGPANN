import sys
import numpy as np 
import pandas as pd

#def addClassColumns(row):
#    classRing = row['Rings']
#    for i in range(1, 30):
#        row.app

filename = "C:\\Users\\bruno\\CLionProjects\\P-CGPDE\\datasets\\abalone.data"

df = pd.read_csv(filename)

classColumn = df['Rings']
#print(classColumn)

df = df.drop('Rings', 1)

for i in range(1,30):
    df[str(i)] = 0

#print(classColumn.values)

for (idx, row) in df.iterrows():
    df.at[idx,str(classColumn[idx])] = 1

print(df)

df.to_csv('C:\\Users\\bruno\\CLionProjects\\P-CGPDE\\datasets\\new_abalone', ' ', index=False)