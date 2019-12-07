import networkx as nx

from collections import namedtuple
from mxnet import autograd, nd, init
from mxnet import gluon
from mxnet.gluon import nn, loss as gloss

import numpy as np
import pandas as pd

DataSet = namedtuple('DataSet', field_names=['X_train', 
                                    'y_train', 'X_test', 'y_test', 'network'])

def load_network(filepath=['./data/karate.edgelist',
                            './data/karate.attributes.csv']):
    network = nx.read_edgelist(filepath[0], nodetype=int)
    attributes = pd.read_csv(filepath[1], index_col=['node'])
    # print(attributes)
    for attribute in attributes.columns.values:
        # print(attribute)
        nx.set_node_attributes(network, values=pd.Series(attributes[attribute],
                                               index=attributes.index).to_dict(),
                        name=attribute)

    # print(network.nodes(data=True))
    X_train, y_train = map(np.array, zip(*[
        ([node], data['role'] == 'Administrator')
        for node, data in network.nodes(data=True) 
        if data['role'] in {'Administrator', 'Instructor'}]))

    X_test, y_test = map(np.array, zip(*[
        ([node], data['community'] == 'Administrator')
        for node, data in network.nodes(data=True) 
        if data['role'] == 'Member']))

    return DataSet(X_train, y_train, X_test, y_test, network)

zkc = load_network()
X_train = zkc.X_train.flatten()
y_train = zkc.y_train
X_test = zkc.X_test.flatten()
y_test = zkc.y_test
network = zkc.network
A = nx.to_numpy_matrix(zkc.network)
A = nd.array(A)

class SpectralRule(nn.Block):
    def __init__(self, A, in_units, out_units, activation='relu', **kwargs):
        super().__init__(**kwargs)
        I = nd.eye(*A.shape)
        A_hat = A.copy() + I
        D = nd.sum(A_hat, axis=0)
        D_inv = D**-0.5
        A_hat = D_inv * A_hat * D_inv
        self.in_units, self.out_units = in_units, out_units
        with self.name_scope():
            self.A_hat = self.params.get_constant('A_hat', A_hat)
            self.W = self.params.get('W', shape=(self.in_units, self.out_units))
            if activation == 'identity':
                self.activation = lambda X: X
            else:
                self.activation = nn.Activation(activation)
        
    def forward(self, X):
        aggregate = nd.dot(self.A_hat.data(), X)
        propagate = self.activation(nd.dot(aggregate, self.W.data()))
        return propagate

class LogisticRegressor(nn.Block):
    def __init__(self, in_units, **kwargs):
        super().__init__(**kwargs)
        self.in_units = in_units
        with self.name_scope():
            self.w = self.params.get('w', shape=(self.in_units, 1))
            self.b = self.params.get('b', shape=(1,))
    
    def forward(self, X):
        y = nd.dot(X, self.w.data()) + self.b.data()
        return nd.sigmoid(y)

def bulid_features(A, X):
    hidden_layers = [(4, 'tanh'), (2, 'tanh')]
    in_units = X.shape[1]
    features = nn.Sequential()
    for layer_size, activation_func in hidden_layers:
        feature = SpectralRule(A, in_units=in_units, out_units=layer_size, 
                                activation=activation_func)
        features.add(feature)
        in_units = layer_size
    return features, in_units

def build_model(A, X):
    model = nn.Sequential()
    with model.name_scope():
        features, out_units = bulid_features(A, X)
        model.add(features)
        calssifier = LogisticRegressor(out_units)
        model.add(calssifier)
    model.initialize(init.Uniform(1))
    return model, features

X_1 = nd.eye(*A.shape)
model_1, features_1 = build_model(A, X_1)

model_1(X_1)

def train(model, features, X, X_train, y_train, epochs):

    cross_entropy = gloss.SigmoidBinaryCrossEntropyLoss(from_sigmoid=True)
    trainer = gluon.Trainer(model.collect_params(), 'sgd',
                            {'learning_rate': 0.0006, 'momentum': 1})
    features_representation = [features(X).asnumpy()]

    for e in range(1, epochs+1):
        cum_loss = 0.0
        cum_preds = []
        for i, x in enumerate(X_train):
            y = nd.array(y_train)[i]
            with autograd.record():
                preds = model(X)[x]
                l = cross_entropy(preds, y)
            l.backward()
            trainer.step(1)
            cum_loss += l.asscalar()
            cum_preds.append(preds.asscalar())
        features_representation.append(features(X).asnumpy())

        if (e % (epochs // 10)) == 0:
            print(f'epochs: {e}/{epochs}, -- loss_sum: {cum_loss: .6f}')
            print(cum_preds)
        
    return features_representation

def predict(model, X, node):
    preds = model(X)[node].asnumpy().flatten()
    return np.where(preds >= 0.5, 1, 0)

# features_rep_1 = train(model_1, features_1, X_1, X_train, y_train, epochs=5000)

# preds = predict(model_1, X_1, X_test)

from sklearn.metrics import classification_report
# print(classification_report(y_test, preds))

X_2 = nd.zeros(shape=(A.shape[0], 2))
node_distance_In = nx.shortest_path_length(network, target=33)
node_distance_Am = nx.shortest_path_length(network, target=0)
for node in network.nodes():
    X_2[node][0] = node_distance_Am[node]
    X_2[node][1] = node_distance_In[node]

model_2, features_2 = build_model(A, X_2)
model_2(X_2)
# print(model_2)

features_rep_2 = train(model_2, features_2, X_2, X_train, y_train, epochs=250)
preds_2 = predict(model_2, X_2, X_test)
print(classification_report(y_test, preds_2))
print(y_test, preds_2)
