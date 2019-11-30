import  collections
import d2lzh as d2l
import math
from mxnet import autograd, gluon, nd
from mxnet.gluon import data as gdata, loss as gloss, nn
import random
import sys
import time
import zipfile


# deal with dataset

with zipfile.ZipFile('./data/ptb.zip', 'r') as zin:
    zin.extractall('./data')

with open('./data/ptb/ptb.train.txt', 'r') as f:
    lines = f.readlines()
    raw_dataset = [st.split() for st in lines]

print('# sentences: %d' % len(raw_dataset))
# 生僻词使用<unk>，数字使用N代替
for st in raw_dataset[:3]:
    print('# tokens:', len(st), st[:5])
# tk : token
counter = collections.Counter([tk for st in raw_dataset for tk in st])
counter = dict(filter(lambda x: x[1] >= 5, counter.items()))
# creat words list
idx_to_token = [tk for tk, _ in counter.items()]
token_to_idx = {tk:idx for idx, tk in enumerate(idx_to_token)}
dataset = [[token_to_idx[tk] for tk in st if tk in token_to_idx]
                 for st in raw_dataset]
num_tokens = sum([len(st) for st in dataset])
print('# tokens: %d' % num_tokens)    

# subsample
def discard(idx):
    return random.uniform(0, 1) < 1 - math.sqrt(
            1e-4 / counter[idx_to_token[idx]] * num_tokens)
subsampled_dataset = [[tk for tk in st if not discard(tk)] for st in dataset]
print('# subsampled_dataset:　%d' % sum([len(st) for st in subsampled_dataset]))

def compare_counts(token):
    return '# %s: before=%d, after=%d' % (token, sum(
        [st.count(token_to_idx[token]) for st in dataset]), sum(
        [st.count(token_to_idx[token]) for st in subsampled_dataset]))

compare_counts('the')
compare_counts('join')

# get centers and contexts
def get_centers_and_contexts(dataset, max_window_size):
    centers, contexts = [], []
    for st in dataset:
        if len(st) < 2:
            continue
        centers += st
        for center_i in range(len(st)):
            window_size = random.randint(1, max_window_size)
            indices = list(range(max(0, center_i - window_size),
                            min(len(st), center_i+window_size+1)))
            indices.remove(center_i)
            contexts.append([st[idx] for idx in indices])
    return centers, contexts

tiny_dataset = [list(range(0, 7)), list(range(7, 10))]
print(tiny_dataset)
for center, context in zip(*get_centers_and_contexts(tiny_dataset, 2)):
    print('center', center, 'has context', context)        

all_centers, all_contexts = get_centers_and_contexts(subsampled_dataset, 5)

# get netatives
def get_negatives(all_contexts, sampling_weights, K):
    all_negatives, neg_candidates, i = [], [], 0
    population = list(range(len(sampling_weights)))
    for contexts in all_contexts:
        negatives = []
        while len(negatives) < len(contexts) * K:
            if i == len(neg_candidates):
                i, neg_candidates = 0, random.choices(
                    population, sampling_weights, k=int(1e5))
            neg, i = neg_candidates[i], i + 1
            if neg not in context:
                negatives.append(neg)
        all_negatives.append(negatives)
    return all_negatives

sampling_weights = [counter[w]**0.75 for w in idx_to_token]
all_negatives = get_negatives(all_contexts, sampling_weights, 5)

def batchify(data):
    max_len = max(len(c)+len(n) for _, c, n in data)
    centers, contexts_negatives, masks, labels = [], [], [], []
    for center, context, negative in data:
        cur_len = len(context) + len(negative)
        centers += [center]
        contexts_negatives += [context + negative + [0]*(max_len - cur_len)]
        masks += [[1] * cur_len + [0] * (max_len - cur_len)]
        labels += [[1] * len(context) + [0] * (max_len - len(context))] 
    return (nd.array(centers).reshape((-1, 1)), nd.array(contexts_negatives),
            nd.array(masks), nd.array(labels))

    
batch_size = 512
num_workers = 0 if sys.platform.startswith('win32') else 4
dataset = gdata.ArrayDataset(all_centers, all_contexts, all_negatives)
data_iter = gdata.DataLoader(dataset, batch_size, shuffle=True,
                            batchify_fn=batchify, num_workers=num_workers)
for batch in data_iter:
    for name, data in zip(['centers', 'contexts_negatives', 'masks','labels'],
                         batch):
        print(name, 'shape:', data.shape)
    break

# skip-gram model
embed = nn.Embedding(input_dim=20, output_dim=4)
embed.initialize()
embed.weight
X = nd.array([[1, 2, 3], [4, 5, 6]])
print(embed(X))

X = nd.ones((2, 1, 4))
Y = nd.ones((2, 4, 6))
print(nd.batch_dot(X, Y).shape)

def skip_gram(center, contexts_and_negatives, embed_v, embed_u):
    v = embed_v(center)
    u = embed_u(contexts_and_negatives)
    pred = nd.batch_dot(v, u.swapaxes(1, 2))
    return pred

loss = gloss.SigmoidBinaryCrossEntropyLoss()
pred = nd.array([[1.5, 0.3, -1, 2], [1.1, -0.6, 2.2, 0.4]])
# 标签变量label中的1和0分别代表背景词和噪声词
label = nd.array([[1, 0, 0, 0], [1, 1, 0, 0]])
mask = nd.array([[1, 1, 1, 1], [1, 1, 1, 0]])  # 掩码变量
print(loss(pred, label, mask) * mask.shape[1] / mask.sum(axis=1))

def sigmd(x):
    return -math.log(1 / (1 + math.exp(-x)))

print('%.7f' % ((sigmd(1.5) + sigmd(-0.3) + sigmd(1) + sigmd(-2)) / 4))
print('%.7f' % ((sigmd(1.1) + sigmd(-0.6) + sigmd(-2.2)) / 3))

embed_size = 100
net = nn.Sequential()
net.add(nn.Embedding(input_dim=len(idx_to_token), output_dim=embed_size,sparse_grad=True),
        nn.Embedding(input_dim=len(idx_to_token), output_dim=embed_size,sparse_grad=True))
def train(net, lr, num_epochs):
    ctx = d2l.try_gpu()
    net.initialize(ctx=ctx, force_reinit=True)
    trainer = gluon.Trainer(net.collect_params(), 'adam',
                            {'learning_rate': lr})
    for epoch in range(num_epochs):
        start, l_sum, n = time.time(), 0.0, 0
        for batch in data_iter:
            center, context_negative, mask, label = [
                data.as_in_context(ctx) for data in batch]
            with autograd.record():
                pred = skip_gram(center, context_negative, net[0], net[1])
                # 使用掩码变量mask来避免填充项对损失函数计算的影响
                l = (loss(pred.reshape(label.shape), label, mask) *
                     mask.shape[1] / mask.sum(axis=1))
            l.backward()
            trainer.step(batch_size)
            l_sum += l.sum().asscalar()
            n += l.size
        print('epoch %d, loss %.2f, time %.2fs'
              % (epoch + 1, l_sum / n, time.time() - start))
train(net, 0.005, 15)

def get_similar_tokens(query_token, k, embed):
    W = embed.weight.data()
    x = W[token_to_idx[query_token]]
    # 添加的1e-9是为了数值稳定性
    cos = nd.dot(W, x) / (nd.sum(W * W, axis=1) * nd.sum(x * x) + 1e-9).sqrt()
    topk = nd.topk(cos, k=k+1, ret_typ='indices').asnumpy().astype('int32')
    for i in topk[1:]:  # 除去输入词
        print('cosine sim=%.3f: %s' % (cos[i].asscalar(), (idx_to_token[i])))

get_similar_tokens('china', 3, net[0])