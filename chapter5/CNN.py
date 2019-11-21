from mxnet import autograd, nd
from mxnet.gluon import nn

# 二维互相关操作
def corr2d(X, K):
    h, w = K.shape
    Y = nd.zeros((X.shape[0]-h+1, X.shape[1]-w+1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i+h,j:j+w] * K).sum()
    return Y


X = nd.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
K = nd.array([[0, 1], [2, 3]])
print(corr2d(X, K))


# 二维卷积层
class Conv2D(nn.Block):
    def __init__(self, kernel_size, **kwargs):
        super(Conv2D, self).__init__(**kwargs)
        self.weight = self.params.get('weight', shape=kernel_size)
        self.bias = self.params.get('bias', shape=(1,))

    def forward(self, x):
        return corr2d(x, self.weight.data()) + self.bias.data()

X = nd.ones((6,8))
X[:, 2:6] = 0
print(X)
K = nd.array([[1,-1]])
Y = corr2d(X, K)
print(Y)

# 
conv2d = nn.Conv2D(1, kernel_size=(1, 2))
conv2d.initialize()

X = X.reshape((1, 1, 6, 8))
Y = Y.reshape((1, 1, 6, 7))

for i in range(10):
    with autograd.record():
        Y_hat = conv2d(X)
        l = (Y_hat - Y)**2
    l.backward()
    conv2d.weight.data()[:] -= 0.03 * conv2d.weight.grad()
    if (i +1 ) % 2 == 0:
        print('batch {} ,loss {}'.format(i+1, l.sum().asscalar()))

#print(Conv2D(X))



X = nd.ones((6,8))
X[:, 2:6] = 0
print(X)
K = nd.array([[1,-1]])
Y = corr2d(X, K)
print(Y)

class Conv2D_(nn.Block):
    def __init__(self, channels, kernel_size, **kwargs):
        super(Conv2D_, self).__init__(**kwargs)
        self.weight = self.params.get(
            'weight', shape=(
                channels,
                1,
            ) + kernel_size)
        self.bias = self.params.get('bias', shape=(channels, ))
        self.num_filter = channels
        self.kernel_size = kernel_size

    def forward(self, x):
        return nd.Convolution(
        data=x, weight=self.weight.data(), bias=self.bias.data(), 
        num_filter=self.num_filter, kernel=self.kernel_size)

X = X.reshape((1, 1, 6, 8))
Y = Y.reshape((1, 1, 6, 7))
conv2d = Conv2D_(10, kernel_size=(1, 2))
conv2d.initialize()
for i in range(10):
    with autograd.record():
        Y_hat = conv2d(X)
        l = (Y_hat - Y)**2
    if i % 2 == 1:
        print('batch %d, loss %.3f' % (i, l.sum().asscalar()))
    l.backward()
    conv2d.weight.data()[:] -= 3e-2 * conv2d.weight.grad()


# 填充
from mxnet import autograd, nd
from mxnet.gluon import nn
def compy_conv2d(conv2d, X):
    conv2d.initialize()
    X = X.reshape((1, 1) + X.shape)
    Y = conv2d(X)
    return Y.reshape(Y.shape[2:])

X = nd.random.normal(shape=(8, 8))

conv2d = nn.Conv2D(1, kernel_size=(3, 3), padding=1)
print(compy_conv2d(conv2d, X).shape)

conv2d = nn.Conv2D(1, kernel_size=(5, 3), padding=(2, 1))
print(compy_conv2d(conv2d, X).shape)

# (n -nk + np + ns)/ns
conv2d = nn.Conv2D(1, kernel_size=(5, 3), padding=(2, 1), strides=2)
print(compy_conv2d(conv2d, X).shape)


# 多输入通道卷积
import d2lzh as d2l

def corr2d_multi_in(X, K):
    # 使用add_n函数来相加(位置参数)
    return nd.add_n(*[d2l.corr2d(x, k) for x, k in zip(X, K)])

X = nd.array([[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
              [[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
K = nd.array([[[0, 1], [2, 3]], [[1, 2], [3, 4]]])

print(corr2d_multi_in(X, K))

# 多输出通道卷积
def corr2d_multi_in_out(X, K):
    return(nd.stack(*[corr2d_multi_in(X, k) for k in  K]))

K = nd.stack(K, K+1, K+2)
print(corr2d_multi_in_out(X, K).shape)


# 1*1卷积
def corr2d_multi_in_out_1x1(X, K):
    c_i, h, w = X.shape
    c_o = K.shape[0]

    X = X.reshape((c_i, h*w))
    K = K.reshape((c_o, c_i))
    Y = nd.dot(K, X)
    return(Y.reshape((c_o, h, w)))

X = nd.random.uniform(shape=(3, 3, 3))
K = nd.random.uniform(shape=(2, 3, 1, 1))
Y1 = corr2d_multi_in_out_1x1(X, K)
Y2 = corr2d_multi_in_out(X, K)

print((Y1 - Y2).norm().asscalar() < 1e-6)


# 池化层，多通道时分别对不同通道池化，
def pool2d(X, pool_size, mode='max'):
    p_h, p_w = pool_size
    Y = nd.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j] = X[i: i + p_h, j: j + p_w].max()
            elif mode == 'avg':
                Y[i, j] = X[i: i + p_h, j: j + p_w].mean()
    return Y

X = nd.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(pool2d(X, (2, 2)), pool2d(X, (2, 2), 'avg'))

X = nd.arange(16).reshape(1, 1, 4, 4)
pool2d = nn.MaxPool2D(3)
print(pool2d(X))

pool2d = nn.MaxPool2D(pool_size=(1,2), strides=(1,2), padding=(1,2))
print(pool2d(X))
                
                