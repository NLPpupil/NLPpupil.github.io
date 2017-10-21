---
layout: post
title:  "PyTorch学习笔记：Tensor"
date:   2017-10-02
categories: PyTorch
---

**torch.Tensor**是含有同一种类型数据的多维矩阵。

**torch.Tensor**构造器默认是**torch.FloatTensor**。

加下划线的是在原址上操作，比如**torch.FloatTensor.abs_()**。

张量里的内容可以用索引和分片提取。

构造器有：
- **class torch.Tensor**
- **class torch.Tensor(\*sizes)**
- **class torch.Tensor(size)**
- **class torch.Tensor(sequence)**
- **class torch.Tensor(ndarray)**
- **class torch.Tensor(tensor)**
- **class torch.Tensor(storage)**


跟张量有关的函数：
- **torch.is_tensor(obj)**
- **torch.is_storage(obj)**
- **torch.set_default_tensor_type(t)**
- **torch.numel(input) → int**：返回input的元素总数。
- **torch.eye(n, m=None, out=None)**：返回一个二维张量，对角线是1，其他是0.
- **torch.from_numpy(ndarray) → Tensor**：返回的张量和ndarray共享内存。
- **torch.linspace(start, end, steps=100, out=None) → Tensor**：返回一个一维张量，等分成steps份。
- **torch.ones(\*sizes, out=None) → Tensor**
- **torch.arange(start, end, step=1, out=None) → Tensor**：间隔是step，区间是[start,end)。
- **torch.zeros(\*sizes, out=None) → Tensor**
- **torch.t(input, out=None) → Tensor**：矩阵转置。
- **torch.randn(\*sizes, out=None) → Tensor**：用正态分布随机一个张量。
- **torch.abs(input, out=None) → Tensor**:逐个（element-wise）计算绝对值。
- **torch.add(input, value, out=None)**：为input的每一个元素加上标量value，返回一个新的张量。
- **torch.eq(input, other, out=None) → Tensor**：逐个比较input和other的元素是否相等。
- **torch.equal(tensor1, tensor2) → bool**：如果两个张量大小相同，且每个位置的元素相同，则返回真。
- **torch.dot(tensor1, tensor2) → float**：计算两个张量的点积（内积）。
- **torch.inverse(input, out=None) → Tensor**：矩阵的inverse。
- **torch.matmul(tensor1, tensor2, out=None)**：矩阵乘法。
