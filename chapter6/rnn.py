# 1 生成数据样本
from mxnet import nd
import random
import zipfile

with zipfile.ZipFile('./data/jaychou_lyrics.txt.zip') as zin:
    with zin.open('jaychou_lyrics.txt') as f:
        corpus_chars = f.read().decode('utf-8')

print(corpus_chars[:40])

# 将换行符转换为空格，便于打印
corpus_chars = corpus_chars.replace('\n', ' ').replace('\r', ' ')
# corpus_chars = corpus_chars[:10000]

# 建立字符索引
idx_to_char = list(set(corpus_chars))
char_to_idx = dict([(char, i) for i, char in enumerate(idx_to_char)])
vocab_size = len(char_to_idx)
print('num_vocab: ',vocab_size)
# 打印前20个字符及其对应的索引
corpus_indices = [char_to_idx[char] for char in corpus_chars]
sample = corpus_indices[:20]
print('chars: ', ''.join([idx_to_char[idx] for idx in sample]))
print('indices: ', sample)


# 时序数据的采样

# 随机采样
def data_iter_random(corpus_indices, batch_size, num_steps, ctx=None):
    # 减1
    num_examples = (len(corpus_indices) -1 ) // num_steps
    epoch_size = num_examples // batch_size
    example_indices = list(range(num_examples))
    random.shuffle(example_indices)

    def _data(pos):
        return corpus_indices[pos: pos + num_steps]

    for i in range(epoch_size):
        # 每次读取batch_size个随机样本
        i = i * batch_size
        batch_indices = example_indices[i: i + batch_size]
        X = [_data(j * num_steps) for j in batch_indices]
        Y = [_data(j * num_steps + 1) for j in batch_indices]
        yield nd.array(X, ctx), nd.array(Y, ctx)
my_seq = list(range(30))
for X, Y in data_iter_random(my_seq, batch_size=2, num_steps=6):
    print('X: ', X, '\nY:', Y, '\n')




# 相邻采样
def data_iter_consecutive(corpus_indices, batch_size, num_steps, ctx=None):
    corpus_indices = nd.array(corpus_indices, ctx=ctx)
    data_len = len(corpus_indices)
    batch_len = data_len // batch_size
    corpus_indices = corpus_indices[:batch_size*batch_len].reshape((batch_size, batch_len))
    epochs = (batch_len - 1) // num_steps
    for i in range(epochs):
        i = i * num_steps
        X = corpus_indices[:, i : i + num_steps]
        Y = corpus_indices[:, i+1 : i + 1 + num_steps]
        yield X, Y

for X, Y in data_iter_consecutive(my_seq, batch_size=2, num_steps=6):
    print('X: ', X, '\nY:', Y, '\n')




# 2实现循环神经网络（字符）
import d2lzh as d2l 
import math
from mxnet import nd, autograd
from mxnet.gluon import loss as gloss
import time

(corpus_indices, char_to_idx, idx_to_char, vocab_size) = d2l.load_data_jay_lyrics()
print(nd.one_hot(nd.array([0, 2]), vocab_size))

def to_onehot(X, size):
    return [nd.one_hot(x, size) for x in X.T]

X = nd.arange(10).reshape((2, 5))
Inputs = to_onehot(X, vocab_size)
print(len(Inputs), Inputs[0].shape)

# 初始化模型参数
num_inputs, num_hiddens, num_outputs = vocab_size, 256, vocab_size
ctx = d2l.try_gpu()
print('use: ', ctx)

def get_params():
    def _ones(shape):
        return nd.random.normal(scale=0.01, shape=shape, ctx=ctx)
    #隐藏层参数
    w_xh = _ones((num_inputs, num_hiddens))
    b_h = nd.zeros(num_hiddens, ctx=ctx)
    w_hh = _ones((num_hiddens, num_hiddens))
    #输出层参数
    w_ho = _ones((num_hiddens, num_outputs))
    b_q = nd.zeros(num_outputs, ctx=ctx)

    params = [w_xh, w_hh, b_h, w_ho, b_q]
    for param in params:
        param.attach_grad()

    return params

# 定义模型

def init_rnn_state(batch_size, num_hiddens, ctx):
    return (nd.zeros(shape=(batch_size, num_hiddens), ctx=ctx),)

def rnn(inputs, state, params):
    w_xh, w_hh, b_h, w_ho, b_q = params
    H, = state
    outputs = []
    # inputs和outputs均为num_steps个（batch_size, vocab_size)的矩阵
    for X in inputs:
        H = nd.tanh(nd.dot(X, w_xh) + nd.dot(H, w_hh) + b_h)
        Y = nd.dot(H, w_ho) + b_q
        outputs.append(Y)
    return outputs, (H,)

state = init_rnn_state(X.shape[0], num_hiddens, ctx)
# print(state)
params = get_params()
inputs = to_onehot(X.as_in_context(ctx), vocab_size)
outputs, new_state = rnn(inputs, state, params)

print(len(outputs), outputs[0].shape, new_state[0].shape)

# 预测函数
def predict_rnn(prefix, num_chars, rnn, params, init_rnn_state, num_hiddens,
                vocab_size, ctx, idx_to_char, char_to_idx):
    state = init_rnn_state(1, num_hiddens, ctx)
    output = [char_to_idx[prefix[0]]]
    for t in range(num_chars + len(prefix) - 1):
        X = to_onehot(nd.array([output[-1]], ctx), vocab_size)
        if t == 0:
            print(len(X))
        # 计算输出和更新隐藏
        (Y, state) = rnn(X, state, params)
        if t < len(prefix)-1:
            output.append(char_to_idx[prefix[t+1]])
        else:
            output.append(int(Y[0].argmax(axis=1).asscalar()))
    
    return ''.join([idx_to_char[i] for i in output])

predict_rnn('分开', 10, rnn, params, init_rnn_state, num_hiddens, vocab_size,
            ctx, idx_to_char, char_to_idx)

# 裁剪梯度
def grad_clipping(params, theta, ctx):
    norm = nd.array([0], ctx)
    for param in params:
        norm += (param.grad**2).sum()
    norm = norm.sqrt().asscalar()
    if norm > theta:
        for param in params:
            param.grad[:] *= theta/norm


def train_and_predict_rnn(rnn, get_params, init_rnn_state, num_hiddens,
                          vocab_size, ctx, corpus_indices, idx_to_char,
                          char_to_idx, is_random_iter, num_epochs, num_steps,
                          lr, clipping_theta, batch_size, pred_period,
                          pred_len, prefixes):
    if is_random_iter:
        data_iter_fn = d2l.data_iter_random
    else:
        data_iter_fn = d2l.data_iter_consecutive
    params = get_params()
    loss = gloss.SoftmaxCrossEntropyLoss()

    for epoch in range(num_epochs):
        if not is_random_iter:  # 如使用相邻采样，在epoch开始时初始化隐藏状态
            state = init_rnn_state(batch_size, num_hiddens, ctx)
        l_sum, n, start = 0.0, 0, time.time()
        data_iter = data_iter_fn(corpus_indices, batch_size, num_steps, ctx)
        for X, Y in data_iter:
            if is_random_iter:  # 如使用随机采样，在每个小批量更新前初始化隐藏状态
                state = init_rnn_state(batch_size, num_hiddens, ctx)
            else:  # 否则需要使用detach函数从计算图分离隐藏状态
                for s in state:
                    s.detach()
            with autograd.record():
                inputs = to_onehot(X, vocab_size)
                # outputs有num_steps个形状为(batch_size, vocab_size)的矩阵
                (outputs, state) = rnn(inputs, state, params)
                # 拼接之后形状为(num_steps * batch_size, vocab_size)
                outputs = nd.concat(*outputs, dim=0)
                # Y的形状是(batch_size, num_steps)，转置后再变成长度为
                # batch * num_steps 的向量，这样跟输出的行一一对应
                y = Y.T.reshape((-1,))
                # 使用交叉熵损失计算平均分类误差
                l = loss(outputs, y).mean()
            l.backward()
            grad_clipping(params, clipping_theta, ctx)  # 裁剪梯度
            d2l.sgd(params, lr, 1)  # 因为误差已经取过均值，梯度不用再做平均
            l_sum += l.asscalar() * y.size
            n += y.size

        if (epoch + 1) % pred_period == 0:
            print('epoch %d, perplexity %f, time %.2f sec' % (
                epoch + 1, math.exp(l_sum / n), time.time() - start))
            for prefix in prefixes:
                print(' -', predict_rnn(
                    prefix, pred_len, rnn, params, init_rnn_state,
                    num_hiddens, vocab_size, ctx, idx_to_char, char_to_idx))

num_epochs, num_steps, batch_size, lr, clipping_theta = 250, 35, 32, 1e2, 1e-2
pred_period, pred_len, prefixes = 50, 50, ['分开', '不分开']


train_and_predict_rnn(rnn, get_params, init_rnn_state, num_hiddens,
                      vocab_size, ctx, corpus_indices, idx_to_char,
                      char_to_idx, True, num_epochs, num_steps, lr,
                      clipping_theta, batch_size, pred_period, pred_len,
                      prefixes)
