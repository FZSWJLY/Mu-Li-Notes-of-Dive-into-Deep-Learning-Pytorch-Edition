# 04 数据操作 + 数据预处理【动手学深度学习v2】

## P1 数据操作

### N维数组样例

- N维数组是机器学习和神经网络的主要数据

![在这里插入图片描述](https://img-blog.csdnimg.cn/e0afc5ad70fa4ab29151047c8a84407c.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA56iL5bqP5ZGY5bCP5YuH,size_17,color_FFFFFF,t_70,g_se,x_16#pic_center)

![在这里插入图片描述](https://img-blog.csdnimg.cn/7d1641e2b24e46b78d8096431065a007.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA56iL5bqP5ZGY5bCP5YuH,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)



### 创建数组

- 形状：例如 3 × 4 矩阵
- 每个元素的数据类型：例如32位浮点数
- 每个元素的值，例如全是0，或者随机数

![在这里插入图片描述](https://img-blog.csdnimg.cn/31721561c215444cbb83fd2fd8e299c7.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA56iL5bqP5ZGY5bCP5YuH,size_16,color_FFFFFF,t_70,g_se,x_16#pic_center)


### 访问元素

![在这里插入图片描述](https://img-blog.csdnimg.cn/c16052a58a5b4c6080170cca37ad2268.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA56iL5bqP5ZGY5bCP5YuH,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)


## P2 数据操作实现

### 数据操作

首先，我们导入 `torch`。请注意，虽然它被称为PyTorch，但我们应该导入 `torch` 而不是 `pytorch`

```python
import torch
```

张量表示由一个数值组成的数组，这个数组可能有多个维度

```python
x = torch.arange(12)
x
```

```python
tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])
```

可以通过张量的 `shape` 属性来访问张量的*形状* 和张量中元素的总数

```python
x.shape
```

```python
torch.Size([12])
```

```python
x.numel()
```

```python
12
```

要改变一个张量的形状而不改变元素数量和元素值，可以调用 `reshape` 函数

```python
X = x.reshape(3, 4)
X
```

使用全0、全1、其他常量或者从特定分布中随机采样的数字

```python
torch.zeros((2, 3, 4))
```

```python
tensor([[[0., 0., 0., 0.],
         [0., 0., 0., 0.],
         [0., 0., 0., 0.]],

        [[0., 0., 0., 0.],
         [0., 0., 0., 0.],
         [0., 0., 0., 0.]]])
```

```python
torch.ones((2, 3, 4))
```

```python
tensor([[[1., 1., 1., 1.],
         [1., 1., 1., 1.],
         [1., 1., 1., 1.]],

        [[1., 1., 1., 1.],
         [1., 1., 1., 1.],
         [1., 1., 1., 1.]]])
```

```python
torch.randn(3, 4)
```

```python
tensor([[ 0.2104,  1.4439, -1.3455, -0.8273],
        [ 0.8009,  0.3585, -0.2690,  1.6183],
        [-0.4611,  1.5744, -0.4882, -0.5317]])
```

通过提供包含数值的 Python 列表（或嵌套列表）来为所需张量中的每个元素赋予确定值

```python
torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
```

```python
tensor([[2, 1, 4, 3],
        [1, 2, 3, 4],
        [4, 3, 2, 1]])
```

```python
torch.tensor([[[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]]])
```

```python
tensor([[[2, 1, 4, 3],
         [1, 2, 3, 4],
         [4, 3, 2, 1]]])
```

```python
torch.tensor([[[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]]]).shape
```

```python
torch.Size([1, 3, 4])
```

常见的标准算术运算符（`+`、`-`、`*`、`/` 和 `**`）都可以被升级为按元素运算

```python
x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])
x + y, x - y, x * y, x / y, x**y
```

```
(tensor([ 3.,  4.,  6., 10.]),
 tensor([-1.,  0.,  2.,  6.]),
 tensor([ 2.,  4.,  8., 16.]),
 tensor([0.5000, 1.0000, 2.0000, 4.0000]),
 tensor([ 1.,  4., 16., 64.]))
```

按元素方式应用更多的计算

```python
torch.exp(x)
```

```python
tensor([2.7183e+00, 7.3891e+00, 5.4598e+01, 2.9810e+03])
```

我们也可以把多个张量 *连结*（concatenate） 在一起

```python
X = torch.arange(12, dtype=torch.float32).reshape((3, 4))
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
torch.cat((X, Y), dim=0), torch.cat((X, Y), dim=1)
```

```python
(tensor([[ 0.,  1.,  2.,  3.],
         [ 4.,  5.,  6.,  7.],
         [ 8.,  9., 10., 11.],
         [ 2.,  1.,  4.,  3.],
         [ 1.,  2.,  3.,  4.],
         [ 4.,  3.,  2.,  1.]]),
 tensor([[ 0.,  1.,  2.,  3.,  2.,  1.,  4.,  3.],
         [ 4.,  5.,  6.,  7.,  1.,  2.,  3.,  4.],
         [ 8.,  9., 10., 11.,  4.,  3.,  2.,  1.]]))
```

通过 *逻辑运算符* 构建二元张量

```python
X == Y
```

```python
tensor([[False,  True, False,  True],
        [False, False, False, False],
        [False, False, False, False]])
```

对张量中的所有元素进行求和会产生一个只有一个元素的张量

```python
X.sum()
```

```python
tensor(66.)
```

即使形状不同，我们仍然可以通过调用 *广播机制* （broadcasting mechanism） 来执行按元素操作

```python
a = torch.arange(3).reshape((3, 1))
b = torch.arange(2).reshape((1, 2))
a, b
```

```python
(tensor([[0],
         [1],
         [2]]),
 tensor([[0, 1]]))
```

```python
a + b
```

```python
tensor([[0, 1],
        [1, 2],
        [2, 3]])
```

可以用 `[-1]` 选择最后一个元素，可以用 `[1:3]` 选择第二个和第三个元素

```python
X[-1], X[1:3]
```

```python
(tensor([ 8.,  9., 10., 11.]),
 tensor([[ 4.,  5.,  6.,  7.],
         [ 8.,  9., 10., 11.]]))
```

除读取外，我们还可以通过指定索引来将元素写入矩阵

```python
X[1, 2] = 9
X
```

```python
tensor([[ 0.,  1.,  2.,  3.],
        [ 4.,  5.,  9.,  7.],
        [ 8.,  9., 10., 11.]])
```

为多个元素赋值相同的值，我们只需要索引所有元素，然后为它们赋值

```python
X[0:2, :] = 12
X
```

```python
tensor([[12., 12., 12., 12.],
        [12., 12., 12., 12.],
        [ 8.,  9., 10., 11.]])
```

运行一些操作可能会导致为新结果分配内存

```python
before = id(Y)
Y = Y + X
id(Y) == before
```

```python
False
```

执行原地操作

```python
Z = torch.zeros_like(Y)
print('id(Z):', id(Z))
Z[:] = X + Y
print('id(Z):', id(Z))
```

```python
id(Z): 140452400950336
id(Z): 140452400950336
```

如果在后续计算中没有重复使用 `X`，我们也可以使用 `X[:] = X + Y` 或 `X += Y` 来减少操作的内存开销

```python
before = id(X)
X += Y
id(X) == before
```

```python
True
```

转换为 NumPy 张量

```python
A = X.numpy()
B = torch.tensor(A)
type(A), type(B)
```

```python
(numpy.ndarray, torch.Tensor)
```

将大小为1的张量转换为 Python 标量

```python
a = torch.tensor([3.5])
a, a.item(), float(a), int(a)
```

```python
(tensor([3.5000]), 3.5, 3.5, 3)
```

## P3 数据预处理实现

### 数据预处理

创建一个人工数据集，并存储在csv（逗号分隔值）文件

```python
import os

os.makedirs(os.path.join('..', 'data'), exist_ok=True)
data_file = os.path.join('..', 'data', 'house_tiny.csv')
with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Price\n')
    f.write('NA,Pave,127500\n')
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')
```

从创建的csv文件中加载原始数据集

```python
# 如果没有安装pandas，只需要取消以下行的注释：
# !pip install pandas
import pandas as pd

data = pd.read_csv(data_file)
print(data)
```

如果在Jupyter中，可将`print(data)`写成`data`

```python
   NumRooms Alley   Price
0       NaN  Pave  127500
1       2.0   NaN  106000
2       4.0   NaN  178100
3       NaN   NaN  140000
```

为了处理缺失的数据，典型的方法包括*插值*和*删除*， 这里，我们将考虑插值

```python
inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
inputs = inputs.fillna(inputs.mean())
print(inputs)
```

```python
   NumRooms Alley
0       3.0  Pave
1       2.0   NaN
2       4.0   NaN
3       3.0   NaN
```

对于`inputs`中的类别值或离散值，我们将“NaN”视为一个类别

```python
inputs = pd.get_dummies(inputs, dummy_na=True)
print(inputs)
```

```python
   NumRooms  Alley_Pave  Alley_nan
0       3.0           1          0
1       2.0           0          1
2       4.0           0          1
3       3.0           0          1
```

现在`inputs`和`outputs`中的所有条目都是数值类型，它们可以转换为张量格式

```python
import torch

X, y = torch.tensor(inputs.values), torch.tensor(outputs.values)
X, y
```

```python
(tensor([[3., 1., 0.],
         [2., 0., 1.],
         [4., 0., 1.],
         [3., 0., 1.]], dtype=torch.float64),
 tensor([127500, 106000, 178100, 140000]))
```

## P4 数据操作 QA

```python
a = torch.arange(12)
b = a.reshape((3,4))
b[:] = 2
a
```

```python
tensor([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
