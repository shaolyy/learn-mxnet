from networkx import read_edgelist, set_node_attributes
from collections import namedtuple
from pandas import read_csv, Series
import numpy as np

import networkx as nx
from mxnet.ndarray import array
from mxnet import ndarray as nd, initializer, autograd
from mxnet.gluon import nn, loss as gloss
from mxnet import gluon

DataSet = namedtuple('DataSet', field_names=[
                  'X_train', 'y_train', 'X_test', 'y_test', 'network'])
network = read_edgelist('./data/karate.edgelist', nodetype=int)
attributes = read_csv('./data/karate.attributes.csv', index_col=['node'])

for attribute in attributes.columns.values:
    print(attribute)
    set_node_attributes(network, values=Series(attributes[attribute],
                                               index=attributes.index).to_dict(),
                        name=attribute)

Series(attributes[attribute],index=attributes.index).to_dict()

for attribute in attributes.columns.values:
    set_node_attributes(
        network,
        values=Series(attributes[attribute],
                        index=attributes.index).to_dict(),
        name=attribute
    )


attributes
X_train, y_train = map(np.array, zip(*[
    ([node], data['role'] == 'Administrator') 
    for node, data in network.nodes(data=True)
    if data['role'] in {'Administrator', 'Instructor'}
]))

X_test, y_test = map(np.array, zip(*[
    ([node], data['community'] == 'Administrator') 
    for node, data in network.nodes(data=True)
    if data['role'] == 'Member' 
]))

zkc = DataSet(X_train, y_train, X_test, y_test, network)

A = nx.to_numpy_matrix(zkc.network)
A = nd.array(A)

X_train = zkc.X_train.flatten()
y_train = zkc.y_train
X_test = zkc.X_test.flatten()
y_test = zkc.y_test

y_test

class SpectralRule(nn.HybridBlock):
    def __init__(self, A, in_units, out_units, activation='relu', **kwargs):
        super().__init__(**kwargs)
        I = nd.eye(*A.shape)
        A_hat = A.copy() + I
        D = nd.sum(A_hat, axis=0)
        D_inv = D**-0.5
        D_inv = nd.diag(D_inv)
        A_hat = D_inv * A_hat * D_inv
        self.in_units, self.out_units = in_units, out_units
        with self.name_scope():
            self.A_hat = self.params.get_constant('A_hat', A_hat)
            self.W = self.params.get('W', shape=(self.in_units, self.out_units))
            if activation == 'identity':
                self.activation = lambda X: X
            else:
                self.activation = nn.Activation(activation)
        
    def hybrid_forward(self, F, X, A_hat, W):
        aggregate = F.dot(A_hat, X)
        propagate = self.activation(F.dot(aggregate, W))
        return propagate

class LogisticRegressor(nn.HybridBlock):
    def __init__(self, in_units, **kwargs):
        super().__init__(**kwargs)
        with self.name_scope():
            self.w = self.params.get('w', shape=(1, in_units))
            self.b = self.params.get('b', shape=(1, 1))
        
    def hybrid_forward(self, F, X, w, b):
        b = F.broadcast_axes(b, axis=(0, 1), size=(34, 1))
        y = F.dot(X, w, transpose_b=True) + b

        return F.sigmoid(y)

def build_features(A, X):
    hidden_layer_specs = [(4, 'tanh'), (2, 'tanh')]
    in_units = X.shape[1]
    features = nn.HybridSequential()
    with features.name_scope():
        for i, (layer_size, activation_func) in enumerate(hidden_layer_specs):
            layer = SpectralRule(A, in_units=in_units, out_units=layer_size,
                                 activation=activation_func)
            features.add(layer)
            in_units = layer_size
    return features, in_units

def build_model(A, X):
    model = nn.HybridSequential()
    in_units = X.shape[1]
    
    with model.name_scope():
        features, out_units = build_features(A, X)
        model.add(features)

        classifier = LogisticRegressor(out_units)
        model.add(classifier)
    model.hybridize()
    model.initialize(initializer.Uniform(1))
    return model, features

X_1 = I = nd.eye(*A.shape)
model_1, features_1 = build_model(A, X_1)
model_1(X_1)

def train(model, features, X, X_train, y_train, epochs):
    cross_entropy = gloss.SigmoidBinaryCrossEntropyLoss(from_sigmoid=True)
    trainer = gluon.Trainer(model.collect_params(), 'sgd', 
                            {'learning_rate':0.001, 'momentum':1})
    feature_representations = [features(X).asnumpy()]

    for e in range(1, epochs+1):
        cum_loss = 0
        cum_preds = []

        for i, x in enumerate(X_train):
            y = nd.array(y_train)[i]
            with autograd.record():
                preds = model(X)[x]
                loss = cross_entropy(preds, y)
            loss.backward()
            trainer.step(1)
            cum_loss += loss.asscalar()
            cum_preds += [preds.asscalar]

        feature_representations.append(features(X).asnumpy())

        if (e % (epochs // 10)) == 0:
            print(f"Eopch {e}/ {epochs} -- Loss: {cum_loss: .4f}")
            print(cum_preds)
    return feature_representations

def predict(model, X, nodes):
    preds = model(X)[nodes].asnumpy().flatten()
    return np.where(preds >= 0.5, 1, 0)

from sklearn.metrics import classification_report
feature_representations_1 = train(model_1, features_1, X_1, X_train, y_train,
                                    epochs=5000)
y_pred_1 = predict(model_1, X_1, X_test)

print(classification_report(y_test, y_pred_1))

