import pandas as pd

df = pd.read_csv('facebook.csv')
df = df.drop('weight', axis = 1)

edgeList = []

for index, row in df.iterrows():
    edgeList.append((row[0],row[1]))