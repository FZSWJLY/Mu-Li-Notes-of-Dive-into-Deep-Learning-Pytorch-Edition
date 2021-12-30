# 05 线性代数【动手学深度学习v2】

## P1 线性代数

### 标量

- 简单操作 

$$
c = a + b
$$

$$
c = a · b
$$

$$
c = sin a
$$

- 长度

$$
|a| = \begin{cases}
 & a \text{ if } a > 0 \\ 
 & - a \text{ otherwise }  
\end{cases}
$$

$$
| a + b | ≤ | a | + | b |
$$

$$
| a  · b | = | a | · | b |
$$

### 向量

- 简单操作

  ​                                                                $\vec{c} = \vec{a} + \vec{b}$ where $\vec{c_{i}} = \vec{a_{i}} + \vec{b_{i}}$

​                                                                             $\vec{c} = α \vec{b}$ where $ α \vec{b_{i}}$

​                                                                         $\vec{c} = sin \vec{a}$ where $\vec{c_{i}} = sin \vec{a_{i}}$

- 长度

$$
\left \| \vec{a} \right \|_{2}=\left [ \sum_{i=1}^{m} \vec{a_{i}}^{2}\right ]^{\frac{1}{2}}
$$

​                                                                                $|| \vec{a} || ≥ 0$ for all $\vec{a}$
$$
|| \vec{a} + \vec{b} || ≤ || \vec{a} || + || \vec{b} ||
$$

$$
|| \vec{a} · \vec{b} || = || \vec{a} || · || \vec{b} ||
$$

![在这里插入图片描述](https://img-blog.csdnimg.cn/06149933456c4755b095601985b65c44.png#pic_center)

$$
\vec{c} = \vec{a} + \vec{b}
$$
![在这里插入图片描述](https://img-blog.csdnimg.cn/7dbe9a9509724edb9051e8d4e8064cd9.png#pic_center)

$$
\vec{c} = α · \vec{b}
$$

- 点乘

$$
\vec{a}^{T} \vec{b} = \sum_{i}^{}\vec{a_{i}} \vec{b_{i}}
$$

- 正交

$$
\vec{a}^{T} \vec{b} = \sum_{i}^{}\vec{a_{i}} \vec{b_{i}} = 0
$$

![在这里插入图片描述](https://img-blog.csdnimg.cn/2fb0c74ffe884ca9b3e62ef2a9ab8015.png#pic_center)


### 矩阵

- 简单操作

​                                                                    $C = A + B$ where $C_{ij} = A_{ij} + B_{ij}$

​                                                                       $C = α · B$ where $C_{ij} = α B_{ij}$

​                                                                       $C = sin A $ where $C_{ij} = sin A_{ij}$

- 乘法（矩阵乘以向量）

​                                                                             $ c = A b $ where $ c_{i} = \sum_{j}^{} A_{i j} b_{j}$

![在这里插入图片描述](https://img-blog.csdnimg.cn/0d5c239fbb174083add08719fb21fc3b.png#pic_center)


![在这里插入图片描述](https://img-blog.csdnimg.cn/00b3eac68f654a1ca9ae0effeff0a1a3.png#pic_center)


- 乘法（矩阵乘以矩阵）

​                                                                           $ C = A B $ where $ C_{i k} = \sum_{j}^{} A_{i j} B_{j k}$

![在这里插入图片描述](https://img-blog.csdnimg.cn/9fddcba60a0745b9a890750c58cc7a8b.png#pic_center)


- **范数**

​                                                                             $ c = A · b $ hence $|| c || ≤ || A || · || b ||$

- 取决于如何衡量 b 和 c 的长度
- **常见范数**
- 矩阵范数：最小的满足上面公式的值
- Frobenius范数

$$
\left \| A \right \|_{Frob}=\left [ \sum_{i j}^{} A_{i j}^{2}\right ]^{\frac{1}{2}}
$$

- **特征向量和特征值**
- 不被矩阵改变方向的向量

![在这里插入图片描述](https://img-blog.csdnimg.cn/401a0d5e4dd04a628f98628fd0571902.png#pic_center)


- 对称矩阵总是可以找到特征向量

### 特殊矩阵

- 对称与反对称

​                                                                            $A_{i j} = A_{j i}$ and $A_{i j} = -A_{j i}$

![在这里插入图片描述](https://img-blog.csdnimg.cn/5f78935fb27f4cd2a40b358b677277f1.png#pic_center)


- 正定

​                                                               $||x||^{2}=x^{T}x≥0$ generalizes to $x^{T}Ax≥0$

- **正交矩阵**
- 所有行都互相正交
- 所有行都有单位长度 $U$ with $\sum_{j}^{} U_{i j}U_{ki}=δ_{ik}$
- 可以写成$UU^{T}=1$
- **置换矩阵**

$P$ where $P_{ij}=1$ if and only if $j=\pi (i)$

- 置换矩阵是正交矩阵

## P2 线性代数实现

### 线性代数

标量由只有一个元素的张量表示

```python
import torch

x = torch.tensor([3.0])
y = torch.tensor([2.0])

x + y, x * y, x / y, x**y
```

```python
(tensor([5.]), tensor([6.]), tensor([1.5000]), tensor([9.]))
```

你可以将向量视为标量值组成的列表

```python
x = torch.arange(4)
x
```

```python
tensor([0, 1, 2, 3])
```

通过张量的索引来访问任一元素

```python
x[3]
```

```python
tensor(3)
```

访问张量的长度

```python
len(x)
```

```python
4
```

只有一个轴的张量，形状只有一个元素

```python
x.shape
```

```python
torch.Size([4])
```

通过指定两个分量mm和 nn来创建一个形状为m×nm×n的矩阵

```python
A = torch.arange(20).reshape(5, 4)
A
```

```python
tensor([[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11],
        [12, 13, 14, 15],
        [16, 17, 18, 19]])
```

矩阵的转置

```python
A.T
```

```python
tensor([[ 0,  4,  8, 12, 16],
        [ 1,  5,  9, 13, 17],
        [ 2,  6, 10, 14, 18],
        [ 3,  7, 11, 15, 19]])
```

*对称矩阵*（symmetric matrix）$A$等于其转置：$A=A^{⊤}$

```python
B = torch.tensor([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
B
```

```python
tensor([[1, 2, 3],
        [2, 0, 4],
        [3, 4, 5]])
```

```python
B == B.T
```

```python
tensor([[True, True, True],
        [True, True, True],
        [True, True, True]])
```

就像向量是标量的推广，矩阵是向量的推广一样，我们可以构建具有更多轴的数据结构

```python
X = torch.arange(24).reshape(2, 3, 4)
X
```

```python
tensor([[[ 0,  1,  2,  3],
         [ 4,  5,  6,  7],
         [ 8,  9, 10, 11]],

        [[12, 13, 14, 15],
         [16, 17, 18, 19],
         [20, 21, 22, 23]]])
```

给定具有相同形状的任意两个张量，任何按元素二元运算的结果都将是相同形状的张量

```python
A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
B = A.clone()
A, A + B
```

```python
(tensor([[ 0.,  1.,  2.,  3.],
         [ 4.,  5.,  6.,  7.],
         [ 8.,  9., 10., 11.],
         [12., 13., 14., 15.],
         [16., 17., 18., 19.]]),
 tensor([[ 0.,  2.,  4.,  6.],
         [ 8., 10., 12., 14.],
         [16., 18., 20., 22.],
         [24., 26., 28., 30.],
         [32., 34., 36., 38.]]))
```

两个矩阵的按元素乘法称为*哈达玛积*（Hadamard product）（数学符号$⊙$）

```python
A * B
```

```python
tensor([[  0.,   1.,   4.,   9.],
        [ 16.,  25.,  36.,  49.],
        [ 64.,  81., 100., 121.],
        [144., 169., 196., 225.],
        [256., 289., 324., 361.]])
```

```python
a = 2
X = torch.arange(24).reshape(2, 3, 4)
a + X, (a * X).shape
```

```python
(tensor([[[ 2,  3,  4,  5],
          [ 6,  7,  8,  9],
          [10, 11, 12, 13]],
 
         [[14, 15, 16, 17],
          [18, 19, 20, 21],
          [22, 23, 24, 25]]]),
 torch.Size([2, 3, 4]))
```

计算其元素的和

```python
x = torch.arange(4, dtype=torch.float32)
x, x.sum()
```

```python
(tensor([0., 1., 2., 3.]), tensor(6.))
```

表示任意形状张量的元素和

```python
A.shape, A.sum()
```

```python
(torch.Size([5, 4]), tensor(190.))
```



```python
A = torch.arange(20 * 2).reshape(2, 5, 4)
A.shape, A.sum()
```

```python
(torch.Size([2, 5, 4]), tensor(780))
```

指定张量沿哪一个轴来通过求和降低维度 

```python
A_sum_axis0 = A.sum(axis=0)
A_sum_axis0, A_sum_axis0.shape
```

```python
(tensor([40., 45., 50., 55.]), torch.Size([4]))
```

```python
A_sum_axis1 = A.sum(axis=1)
A_sum_axis1, A_sum_axis1.shape
```

```python
(tensor([ 6., 22., 38., 54., 70.]), torch.Size([5]))
```

```python
A.sum(axis=[0, 1])
```

```python
tensor(190.)
```

一个与求和相关的量是*平均值*（mean或average）

```python
A.mean(), A.sum() / A.numel()
```

```python
(tensor(9.5000), tensor(9.5000))
```

```python
A.mean(axis=0), A.sum(axis=0) / A.shape[0]
```

```python
(tensor([ 8.,  9., 10., 11.]), tensor([ 8.,  9., 10., 11.]))
```

计算总和或均值时保持轴数不变

```python
sum_A = A.sum(axis=1, keepdims=True)
sum_A
```

```python
tensor([[ 6.],
        [22.],
        [38.],
        [54.],
        [70.]])
```

通过广播将`A`除以`sum_A`

```python
A / sum_A
```

```python
tensor([[0.0000, 0.1667, 0.3333, 0.5000],
        [0.1818, 0.2273, 0.2727, 0.3182],
        [0.2105, 0.2368, 0.2632, 0.2895],
        [0.2222, 0.2407, 0.2593, 0.2778],
        [0.2286, 0.2429, 0.2571, 0.2714]])
```

某个轴计算`A`元素的累积总和

```python
A.cumsum(axis=0)
```

```python
tensor([[ 0.,  1.,  2.,  3.],
        [ 4.,  6.,  8., 10.],
        [12., 15., 18., 21.],
        [24., 28., 32., 36.],
        [40., 45., 50., 55.]])
```

点积是相同位置的按元素乘积的和

```python
y = torch.ones(4, dtype=torch.float32)
x, y, torch.dot(x, y)
```

```python
(tensor([0., 1., 2., 3.]), tensor([1., 1., 1., 1.]), tensor(6.))
```

我们可以通过执行按元素乘法，然后进行求和来表示两个向量的点积

```python
torch.sum(x * y)
```

```python
tensor(6.)
```

矩阵向量积$Ax$是一个长度为$m$的列向量，其第$i$个元素是点积${a_{i}}^{⊤}x$

```python
A.shape, x.shape, torch.mv(A, x)
```

```python
(torch.Size([5, 4]), torch.Size([4]), tensor([ 14.,  38.,  62.,  86., 110.]))
```

我们可以将矩阵-矩阵乘法$AB$看作是简单地执行$m$次矩阵-向量积，并将结果拼接在一起，形成一个$n×m$矩阵

```python
B = torch.ones(4, 3)
torch.mm(A, B)
```

```python
tensor([[ 6.,  6.,  6.],
        [22., 22., 22.],
        [38., 38., 38.],
        [54., 54., 54.],
        [70., 70., 70.]])
```

$L_{2}$*范数*是向量元素平方和的平方根：       
$$
\left \| x \right \|_{2}=\sqrt{\sum_{i=1}^{n}x_{i}^{2}}
$$

```python
u = torch.tensor([3.0, -4.0])
torch.norm(u)
```

```python
tensor(5.)
```

$L_{1}$范数，它表示为向量元素的绝对值之和：
$$
\left \| x \right \|_{1}=\sum_{i=1}^{n}|x_{i}|
$$

```python
torch.abs(u).sum()
```

```python
tensor(7.)
```

矩阵 的*弗罗贝尼乌斯范数*（Frobenius norm）是矩阵元素平方和的平方根：
$$
\left \| X \right \|_{F}=\sqrt{\sum_{i=1}^{m}\sum_{j=1}^{n}x_{ij}^{2}}
$$

```python
torch.norm(torch.ones((4, 9)))
```

```python
tensor(6.)
```

## P3 按特定轴求和

![在这里插入图片描述](https://img-blog.csdnimg.cn/1a0b20fd165647358f7afcde00cb9a64.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA56iL5bqP5ZGY5bCP5YuH,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)


```python
import torch

a = torch.ones((2, 5, 4))
a.shape
```

```python
torch.Size([2, 5, 4])
```

```python
a.sum().shape
```

```python
torch.Size([])
```

```python
a.sum(axis = 1).shape
```

```python
torch.Size([2, 4])
```

```python
a.sum(axis = 1)
```

```python
tensor([[5., 5., 5., 5.],
        [5., 5., 5., 5.]])
```

```python
a.sum(axis = 0).shape
```

```python
torch.Size([5, 4])
```

```python
a.sum(axis = [0, 2]).shape
```

```python
torch.Size([5])
```

```python
a.sum(axis = 1,keepdims = True).shape
```

```python
torch.Size([2, 1, 4])
```

```python
a.sum(axis = [0, 2],keepdims = True).shape
```

```python
torch.Size([1, 5, 1])
```


