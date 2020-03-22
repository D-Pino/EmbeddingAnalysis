import igraph as ig

#get edgelist from somewhere

g = ig.Graph()
g.add_vertices(4039)
g.add_edges(edgeList)

print(g.modularity(km.labels,None))
