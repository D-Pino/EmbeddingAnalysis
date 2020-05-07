import pandas as pd
import pickle as pik

df = pd.read_csv('musae_RU_edges.csv')
df = df.drop('weight', axis = 1)

edgeList = []

for index, row in df.iterrows():
    edgeList.append((row[0],row[1]))

with open('twitchRUEdgeList.txt', 'wb') as f:
    pik.dump(edgeList,f)



