import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# load karate network 
zkc = nx.karate_club_graph()
order = sorted(list(zkc.nodes()))

# input parameters
A = nx.to_numpy_matrix(zkc, nodelist=order)
I = np.eye(zkc.number_of_nodes())
A_hat = A + I
D_hat = np.array(np.sum(A_hat, axis=1))
D_hat = [x[0] for x in D_hat]
D_hat = np.matrix(np.diag(D_hat))

# initializing weight
# Normal Distribution or Gaussian Distribution
W_1 = np.random.normal(loc=0, scale=1, size=(zkc.number_of_nodes(), 4))  
W_2 = np.random.normal(loc=0, size=(W_1.shape[1], 2))
# loc is Mean (“centre”) of the distribution
# scale is Standard deviation (spread or “width”) of the distribution
# size is Output shape

# GCN Model
def relu(x):
    return np.maximum(x, 0)

def gcn_layer(A, D, X, W):
    return relu(D**-1*A*X*W)

H_1 = gcn_layer(A_hat, D_hat, I, W_1)
H_2 = gcn_layer(A_hat, D_hat, H_1, W_2)
output = H_2

# get feature representations of nodes
feature_representations = {node: np.array(output)[node] for node in zkc.nodes()}

# visualizing the features
plt.figure(figsize=(7, 5), dpi=180)  # set the size of the figure
pos = feature_representations
for x in zkc.nodes():
    if zkc.nodes[x]['club'] == 'Mr. Hi':
        nx.draw_networkx_nodes(zkc, pos, [x], node_size = 200,
                node_color = '#1f77b4', alpha=1)
        
    else:
        nx.draw_networkx_nodes(zkc, pos, [x], node_size = 200,
                node_color = '#ff7f0e', alpha=1) 

labels = {}
labels[0] = r'A'
labels[33] = r'I'
nx.draw_networkx_labels(zkc, pos, labels, font_size=8)
# plt.axis('off')
plt.show()