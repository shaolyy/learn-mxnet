import d2lzh as d2l
from mxnet import gluon, autograd, nd
from mxnet.gluon import loss as gloss, data as gdata, nn
import numpy as numpy 
import pandas as  pd

# 读取数据
def load_data(file_path= ['./data/house_prices/train.csv','./data/house_prices/test.csv']):

    train = pd.read_csv(file_path[0])
    test = pd.read_csv(file_path[1])

    all_features = pd.concat((train.iloc[:,1:-1], test.iloc[:,1:-1]),axis=0,sort=False)

    numeric_feas = all_features.dtypes[all_features.dtypes != 'object'].index
    all_features[numeric_feas] = all_features[numeric_feas].apply(lambda x: (x - x.mean() / (x.std())))
    all_features[numeric_feas] = all_features[numeric_feas].fillna(0)

    all_features = pd.get_dummies(all_features, dummy_na=True)

    n_train = train.shape[0]

    train_feas = nd.array(all_features[:n_train].values)
    test_feas = nd.array(all_features[n_train:].values)

    train_labels = nd.array(train.SalePrice.values).reshape((-1,1))

    return train_feas, train_labels, test_feas, test

# 建立模型

# 损失函数
loss = gloss.L2Loss()

# 建立网络模型
def get_net():
    net = nn.Sequential()
    net.add(nn.Dense(30, activation='relu'))
    net.add(nn.Dense(1))
    net.initialize()

    return net

# log 均方误差
def rmse_log(net, features, labels):
    pred = net(features)
    cliped_pred = nd.clip(pred, 1, float('inf'))
    l = nd.sqrt(2 * loss(nd.log(cliped_pred),nd.log(labels)).mean())

    return l

# 训练函数
def train_mo(net, train_feas, train_labels, test_feas, test_labels, 
            num_epochs, learning_rate, weight_decay, batch_size):
        train_ls, test_ls = [], []
        train_iter = gdata.DataLoader(gdata.ArrayDataset(train_feas, train_labels), batch_size=batch_size, shuffle=True)
        #net = get_net()
        trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate':learning_rate, 'wd':weight_decay})
        for _ in range(num_epochs):
            for x, y in train_iter:
                with autograd.record():
                    l = loss(net(x), y)
                l.backward()
                trainer.step(batch_size)
            train_ls.append(rmse_log(net,train_feas, train_labels))
            if test_labels is not None:
                test_ls.append(rmse_log(net, test_feas, test_labels))
        
        return train_ls, test_ls

# 生成K折检验数据
def get_k_fold_data(k, i, X, y):
    assert k > 1
    num_fold = y.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(i*num_fold, (i+1)*num_fold)
        X_part, y_part = X[idx], y[idx]
        if i == j:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = nd.concat(X_train, X_part, dim=0)
            y_train = nd.concat(y_train, y_part, dim=0)
    
    return X_train, y_train, X_valid, y_valid

# 进行K折训练
def k_fold(k, x_train, y_train, num_epochs, learning_rate, weight_decay, batch_size):
    train_ls_sum, test_ls_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k, i, x_train, y_train)
        net = get_net()
        train_ls, test_ls = train_mo(net, *data, num_epochs, learning_rate, weight_decay, batch_size)
        train_ls_sum += train_ls[-1]
        test_ls_sum += test_ls[-1]
        #print(len(data[1]))
        print('在第{}折训练中，训练误差为：{:g}，验证误差为：{:g}'.format(i+1, train_ls[-1].asscalar(), test_ls[-1].asscalar()))
    print('k折平均训练误差{}，平均验证误差{}'.format(train_ls_sum.asscalar() / k, test_ls_sum.asscalar() / k))
    return train_ls_sum / k, test_ls_sum / k


def train_and_predic(train_feas, train_labels, test_feas, test_data, 
            num_epochs, learning_rate, weight_decay, batch_size,save_path='./data/house_prices/submission.csv'):
    net = get_net()
    train_ls, _ = train_mo(net, train_feas, train_labels, None, None, num_epochs, learning_rate, weight_decay, batch_size)
    print('训练误差为{}'.format(train_ls[-1].asscalar()))

    #d2l.semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'rmse')

    test_pred = net(test_feas).asnumpy()
    test_data['SalePrice'] = pd.Series(test_pred.reshape(1, -1)[0])
    submission = pd.concat([test_data['Id'],test_data['SalePrice']], axis=1)

    submission.to_csv(save_path, index=False)




if __name__ == "__main__":
    k, num_epochs, lr, weight_decay, batch_size = 5, 300, 0.01, 0, 64
    
    train_features, train_labels, test_features, test_data = load_data()
    
    train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr,
                          weight_decay, batch_size)
    train_and_predic(train_features, train_labels, test_features, test_data, num_epochs, lr, weight_decay, batch_size)
    
    


    

    