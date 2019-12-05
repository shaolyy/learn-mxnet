class SpectralRule(HybridBlock):
    def __init__(self,
                 A, in_units, out_units,
                 activation, **kwargs):
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
            self.W = self.params.get(
                'W', shape=(self.in_units, self.out_units)
            )
            if activation == 'ident':
                self.activation = lambda X: X
            else:
                self.activation = Activation(activation)    

	def hybrid_forward(self, F, X, A_hat, W):
        aggregate = F.dot(A_hat, X)
        propagate = self.activation(
            F.dot(aggregate, W))
        return propagate

def build_model(A, X):
    model = HybridSequential()    
    
    with model.name_scope():
        features = build_features(A, X)
        model.add(features)        
        classifier = LogisticRegressor()
        model.add(classifier)        
        model.initialize(Uniform(1))    
      
    return model, features

def train(model, features, X, X_train, y_train, epochs):
    cross_entropy = SigmoidBinaryCrossEntropyLoss(from_sigmoid=True)
    trainer = Trainer(
        model.collect_params(), 'sgd',
        {'learning_rate': 0.001, 'momentum': 1})    
    
    feature_representations = [features(X).asnumpy()]    

	for e in range(1, epochs + 1):
        for i, x in enumerate(X_train):
            y = array(y_train)[i]
            with autograd.record():
                pred = model(X)[x] # Get prediction for sample x
                loss = cross_entropy(pred, y)
            loss.backward()
            trainer.step(1)        
		feature_representations.append(features(X).asnumpy())    

	return feature_representations