---
layout: post
title:  "用Chainer实现LSTM语言模型"
date:   2017-10-11
categories: 深度学习 Chainer 自然语言处理
---

训练一个语言模型，输入是一序列的词（上下文），输出一个概率分布来预测下一个词。

训练数据是[Penn Tree Bank](https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.train.txt)数据集，生数据的形式是一句一行。经过        **chainer.datasets.get_ptb_words()** 预处理后，所有句子都连在一起变成一个超长序列，用 **\<eos\>** 表示句子结尾。同时每个词都变成一个整数ID。


下面进入实现环节，完整代码见[Chainer Recurrent Net Language Model](https://github.com/chainer/chainer/tree/master/examples/ptb)。
<h4>RNNForLM</h4>
先定义用于语言模型的RNN类：

```python
# Definition of a recurrent net for language modeling
class RNNForLM(chainer.Chain):

    def __init__(self, n_vocab, n_units):
        super(RNNForLM, self).__init__()
        with self.init_scope():
            self.embed = L.EmbedID(n_vocab, n_units)
            self.l1 = L.LSTM(n_units, n_units)
            self.l2 = L.LSTM(n_units, n_units)
            self.l3 = L.Linear(n_units, n_vocab)

        for param in self.params():
            param.data[...] = np.random.uniform(-0.1, 0.1, param.data.shape)

    def reset_state(self):
        self.l1.reset_state()
        self.l2.reset_state()

    def __call__(self, x):
        h0 = self.embed(x)
        h1 = self.l1(F.dropout(h0))
        h2 = self.l2(F.dropout(h1))
        y = self.l3(F.dropout(h2))
        return y
```

先由一个嵌入映射把词转换成向量，这里每个词是一个独热(one hot)向量，即相应坐标的值是1，其他是0。再由两层LSTM把词向量映射到另一个向量，最后一个仿射变换（全连接神经网络）把该向量映射到概率分布。

整个映射的参数有词嵌入的矩阵，两个LSTM的权重，和仿射变换的权重。初始参数值均匀随机分布。这个RNNForLM链只实现了一步的输入输出，一个序列的输入只需要逐个输入进去。
<h4>迭代器</h4>
然后要自定义一个迭代器。迭代器在数据集中迭代，每一次迭代产生一个minibatch。minibatch是一个链表的样本(examples)。在这个问题里，minibatch是一个(input,target)链表，形如[($x_{1}$,$t_{1}$), ($x_{2}$,$t_{2}$), ... , ($x_{n-1}$,$t_{n-1}$), ($x_{n}$,$t_{n}$)]，$n$是batch_size，$t_{1}$ 是数据集里 $x_{1}$ 的下一个词，$x_{2}$ 是数据集里 $x_{1}$ 之后的len(数据集)/batch_size个词。换句话说，每次调用迭代器，会在数据集里等距离选取batch_size个词作为inputs，然后右移一个单位等距离选取batch_size个词作为targets。下一次调用迭代器的时候，inputs是上一次调用的targets。部分代码如下：

```python
class ParallelSequentialIterator(chainer.dataset.Iterator):

    def __init__(self, dataset, batch_size, repeat=True):
        self.dataset = dataset
        self.batch_size = batch_size  # batch size
        self.epoch = 0
        self.is_new_epoch = False
        self.repeat = repeat
        length = len(dataset)
        self.offsets = [i * length // batch_size for i in range(batch_size)]
        self.iteration = 0
        # use -1 instead of None internally
        self._previous_epoch_detail = -1.

    def __next__(self):
        length = len(self.dataset)
        if not self.repeat and self.iteration * self.batch_size >= length:
            raise StopIteration
        cur_words = self.get_words()
        self._previous_epoch_detail = self.epoch_detail
        self.iteration += 1
        next_words = self.get_words()

        epoch = self.iteration * self.batch_size // length
        self.is_new_epoch = self.epoch < epoch
        if self.is_new_epoch:
            self.epoch = epoch
        return list(zip(cur_words, next_words))
```

<h4>更新器</h4>
更新器完成一步的反向传播和参数更新。RNN里的反向传播是Backpropgation Through Time，或Truncated BPTT。

```python
# Custom updater for truncated BackProp Through Time (BPTT)
class BPTTUpdater(training.StandardUpdater):
    def __init__(self, train_iter, optimizer, bprop_len, device):
        super(BPTTUpdater, self).__init__(train_iter, optimizer, device=device)
        self.bprop_len = bprop_len

    def update_core(self):
        loss = 0
        # When we pass one iterator and optimizer to StandardUpdater.__init__,
        # they are automatically named 'main'.
        train_iter = self.get_iterator('main')
        optimizer = self.get_optimizer('main')

        # Progress the dataset iterator for bprop_len words at each iteration.
        for i in range(self.bprop_len):
            # Get the next batch (a list of tuples of two word IDs)
            batch = train_iter.__next__()
            x, t = self.converter(batch, self.device)
            loss += optimizer.target(chainer.Variable(x), chainer.Variable(t))

        optimizer.target.cleargrads()
        loss.backward()  # Backprop
        loss.unchain_backward()  # Truncate the graph
        optimizer.update()  # Update the parameters
```
Truncated BPTT就是给反向传播设置了一个最大传播长度bprop_len，每bprop_len次累积损失之后，反向传播一下，然后用**unchain_backward()** 切断此刻的loss跟过去的联系，这样下一个bprop_len次累积集齐之后，反向传播的长度只是最近的bprop_len个LSTM展开，不会再理会更早的展开。converter把batch中所有的x连接成一个数组，所有的t连接成一个数组。

---
到这里关键的地方就说完了，整个过程就是先定义网络的组成并为参数随机初始值，然后同时训练词嵌入矩阵和LSTM语言模型。迭代器根据具体需求要自定义，更新器要实现Truncated BPTT，实现Truncated BPTT的方法也很简单，就是设置一个最大反向传播距离，然后每次到了这个距离就切断LSTM展开链，重新开始。
