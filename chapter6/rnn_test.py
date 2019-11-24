import d2lzh as d2l 
import math
from mxnet import autograd, nd
from mxnet.gluon import loss as gloss
import time

(corpus_indices, char_to_idx, idx_to_char, 
    vocab_size) = d2l.load_data_jay_lyrics()

def one_hot(X, size):
    return [nd.one_hot(x, size) for x in X.T]

X = nd.arange(10).reshape((2, 5))
inputs = one_hot(X, vocab_size)
print(len(inputs), inputs[0].shape)

num_inputs, num_hiddens, num_outputs = vocab_size, 256, vocab_size

ctx = d2l.try_gpu()
print('use: ', ctx)

def get_params():
    '''返回初始化的参数'''
    def _one(shape):
        return nd.random.normal(scale=0.01, shape=shape, ctx=ctx)
    w_xh = _one((num_inputs, num_hiddens))
    w_hh = _one((num_hiddens, num_hiddens))
    w_q = _one((num_hiddens, num_outputs))

    b_xh = nd.zeros(num_hiddens, ctx=ctx)
    b_q = nd.zeros(num_outputs, ctx=ctx)

    params = [w_xh, w_hh, b_xh, w_q, b_q]
    for param in params:
        param.attach_grad()
    return params

def init_rnn_state(batch_size, num_hiddens, ctx):
    ''' 返回初始化的隐藏状态'''
    return (nd.zeros(shape=(batch_size, num_hiddens), ctx=ctx),)

def rnn(inputs, state, params):
    '''返回输出和隐藏状态'''
    w_xh, w_hh, b_h, w_q, b_q = params
    H, = state
    outputs = []
    for X in inputs:
        H = nd.relu(nd.dot(X, w_xh) + nd.dot(H, w_hh) + b_h)
        Y = nd.dot(H, w_q) + b_q
        outputs.append(Y)

    return outputs, (H,)

state = init_rnn_state(X.shape[0], num_hiddens, ctx)
params = get_params()

outputs, new_state = rnn(inputs, state, params)
print(len(outputs), outputs[0].shape, new_state[0].shape)

def predict_rnn(prefix, num_chars, rnn, params, init_rnn_state,
                num_hiddens, vocab_size, ctx, idx_to_char, char_to_idx):
    '''返回预测的字符串'''
    state = init_rnn_state(1, num_hiddens, ctx)
    outputs = [char_to_idx[prefix[0]]]
    for i in range(len(prefix) + num_chars - 1):
        x = one_hot(nd.array([outputs[-1]], ctx=ctx), vocab_size)
        # x = one_hot(nd.array([outputs[-1]], ctx=ctx), vocab_size)
        (Y, state) = rnn(x, state, params)
        if i < len(prefix)-1:
            outputs.append(char_to_idx[prefix[i+1]])
        else:
            outputs.append(int(Y[0].argmax(axis=1).asscalar()))
    return ''.join([idx_to_char[idx] for idx in outputs])

predict_rnn('分开', 10, rnn, params, init_rnn_state, num_hiddens, vocab_size,
            ctx, idx_to_char, char_to_idx)


def grad_clipping(params, theta, ctx):
    '''对传入的梯度进行裁剪'''
    norm = nd.array([0], ctx=ctx)
    for param in params:
        norm += ((param.grad)**2).sum()
    norm = norm.sqrt().asscalar()
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm 



def train_and_predict_rnn(rnn, get_params, init_rnn_state, num_hiddens,
                          vocab_size, ctx, corpus_indices, idx_to_char,
                          char_to_idx, is_random_iter, num_epochs, num_steps,
                          lr, clipping_theta, batch_size, pred_period,
                          pred_len, prefixes):
    '''训练模型，并进行预测'''
    if is_random_iter:
        data_iter_fn = d2l.data_iter_random
    else:
        data_iter_fn = d2l.data_iter_consecutive
    params = get_params()
    loss = gloss.SoftmaxCrossEntropyLoss()
    for epoch in range(num_epochs):
        if not is_random_iter:
            state = init_rnn_state(batch_size, num_hiddens, ctx=ctx)
        l_sum, n, start = 0.0, 0, time.time()
        data_iter = data_iter_fn(corpus_indices,batch_size, num_steps, ctx=ctx)
        for X, Y in data_iter:
            if is_random_iter:
                state = init_rnn_state(batch_size, num_hiddens, ctx=ctx)
            else:  # 否则需要使用detach函数从计算图分离隐藏状态
                for s in state:
                    s.detach()
            inputs = one_hot(X, vocab_size)
            with autograd.record():
                (outputs, state) = rnn(inputs, state, params)
                outputs = nd.concat(*outputs, dim=0)
                y = Y.T.reshape((-1,))
                l = loss(outputs, y).mean()
            l.backward()
            grad_clipping(params, clipping_theta, ctx=ctx)
            d2l.sgd(params, lr, 1)
            l_sum += l.asscalar()*y.size
            n += y.size
        
        if (epoch+1) % pred_period == 0:
            print('epoch: %d, perlexity: %f, time %.2f sec' % (
                    epoch + 1, math.exp(l_sum/n), time.time() - start))
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

train_and_predict_rnn(rnn, get_params, init_rnn_state, num_hiddens,
                      vocab_size, ctx, corpus_indices, idx_to_char,
                      char_to_idx, False, num_epochs, num_steps, lr,
                      clipping_theta, batch_size, pred_period, pred_len,
                      prefixes)