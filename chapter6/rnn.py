from mxnet import nd
import random
import zipfile

with zipfile.ZipFile('./data/jaychou_lyrics.txt.zip') as zin:
    with zin.open('jaychou_lyrics.txt') as f:
        corpus_chars = f.read().decode('utf-8')

print(corpus_chars[:40])

# 将换行符转换为空格，便于打印
corpus_chars = corpus_chars.replace('\n', ' ').replace('\r', ' ')
corpus_chars = corpus_chars[:10000]

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
    num_examples = len(corpus_indices) // num_steps
    epoch_size = num_examples // batch_size
    examples_indices = list(range(num_examples))
    random.shuffle(examples_indices)

    def _data(pos):
        return examples_indices[pos:pos+num_steps]

    for epoch in range(epoch_size):
        i = epoch*batch_size
        batch_indices = examples_indices[i:i+batch_size]
        X = [_data(idx*num_steps) for idx in batch_indices]
        Y = [_data(idx*num_steps+1) for idx in batch_indices]

        yield nd.array(X, ctx=ctx), nd.array(Y, ctx=ctx) 

# 相邻采样
def data_iter_consecutive(corpus_indices, batch_size, num_steps, ctx=None):
    pass


