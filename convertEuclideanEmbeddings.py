import pandas as pd

#first, we manually convert to a csv (replace space with comma and save as .csv)
#also manually get rid of the first row with non-essential info
X = pd.read_csv('facebookDeepwalkEmbeddings.csv', sep =',', header = None)
X = X.sort_values(by = 0)
X = X.reset_index()
X = X.drop(["index",0], axis = 1)
embeddings = X.to_numpy()

