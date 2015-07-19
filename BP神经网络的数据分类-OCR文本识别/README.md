
<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

### 1. 数据集说明

本次所用的数据集有 5000 个样本，每个样本对应于 20x20 大小的灰度图像。对应 9-0 共十个数字的手写图像。样本中每个像素都用浮点数表示。在样本数据中，每幅图像都被展开为一个 400 维的向量，构成了样本数据矩阵中的一行。完整的训练数据是一个 5000x400 的矩阵，其每一行为一个样本。另外一个数据用来表示上述样本的标号，为 5000x1 的矩阵，数据中，对应于数字”0”的图像被标记为”10”，而数字”1”到”9”按照其自然顺序被分别标记为”1”到”9”。


### 2. 后向传播算法（BP）

定义误差：
$$
{e_n} = \frac{1}{2}{\left( {y - {W_{21}}H} \right)^2}
$$

1. 网络初始化：
    1. 根据输入输出序列（X，Y）确定：
        + 网络输入层节点数 n（不考虑额外的偏置项为样本长度，如果考虑单元个数需要+1）
        + 隐含层结点数 l
        + 输出层节点数 m
    2. 初始化：
    + 初始化输入层与隐藏层的之间的权值 $w_{j,i},0<j \le l;0<i \le n$，可以组成矩阵 $W_{10}$
    + 初始化隐藏层与输出层之间的权值 $w_{k,j},0<k \le n;0<j \le l$，可以组成矩阵 $W_{21}$
    + 给定学习速率 $\eta$
    + 给定神经元激励函数：$\sigma \left( y \right) = \frac{1}{{1 + {e^{ - y}}}}$ （本文采用）或 $\sigma \left( y \right) = \frac{{{e^y} - {e^{ - y}}}}{{{e^y} + {e^{ - y}}}} = \tanh \left( y \right)$

2. 隐藏层输出计算：
    随机选择一个样本 （x，y），计算隐含层输出 H：
    $${H_{l \times 1}} = \sigma \left( {{W_{10}}x_{n \times 1}} \right)$$
3. 输出层输出计算：
    根据隐藏层的输出 H，计算 BP 神经网络的预测输出 O：
    $${O_{m \times 1}} = {W_{21}}{H_{l \times 1}}$$
4. 误差计算：
    根据网络的预测输出 O 和期望输出 y，计算网络的预测误差 eO：
    $$eO = y_{m \times 1} - O_{m \times 1}$$
5. 对于网络的输出单元，计算它们的误差项：
    $$\begin{array}{l}
\delta _j^{\left( L \right)} &= \frac{{\partial {e_n}}}{{\partial H_j^{\left( L \right)}}} = \left( {{y_i} - {O_i}} \right)\\
{\delta ^{\left( L \right)}} &= \left( {{y_{m \times 1}} - {O_{m \times 1}}} \right)=eO
\end{array}$$
6. 对于网络的隐藏层单元，计算它们的误差项：
$$\begin{array}{l}
\delta _j^{\left( l \right)} &= \sum\limits_{k = 1}^{{d^{\left( {l + 1} \right)}}} {\left( {\left( {\delta _k^{\left( {l + 1} \right)}} \right) * \left( {w_{kj}^{\left( {l + 1} \right)}} \right) * \left( {\sigma '\left( {H_j^{\left( l \right)}} \right)} \right)} \right)} \\
{\delta ^{\left( l \right)}} &= \sigma '\left( {{H^{\left( l \right)}}} \right).*{\left( {W_{21}^T{\delta ^{\left( L \right)}}} \right)_{l \times 1}}
\end{array}$$   
7. 更新每个网络权重：
$$\begin{array}{l}
{w_{ji}} &= {w_{ji}} + \Delta {w_{ji}} = {w_{ji}} + \eta {\delta _j}{x_{ji}}\\
{W_{21}} &= {W_{21}} + \eta {\delta ^{\left( L \right)}} * {H^T}\\
{W_{10}} &= {W_{10}} + \eta {\delta ^{\left( l \right)}}*{x^T}
\end{array}$$

### 3. 后向传播算法-代码
```m
% 取一个样本
oneIn = inputn(:, n);
oneOut = output_train(:, n);

% 隐藏层输出             
hOut = 1 ./ (1 + exp(- W10 * oneIn));

% 输出层输出
yOut = W21 * hOut;

% 计算误差
eOut = oneOut - yOut;     

% 计算输出层误差项 delta2
delta2 = eOut;

% 计算隐藏层误差项 delta1
FI = hOut .* (1 - hOut);
delta1 = (FI .* (W21' * eOut));

% 更新权重
W21 = W21 + eta * delta2 * hOut';
W10 = W10 + eta * delta1 * oneIn';
```

### 4. 实验

从 5000 个样本中随机选择 4000 个进行训练，剩余的 1000个用于测试，采用不同的迭代次数，不同的学习速率参数，分别进行十次独立随机实验，识别的准确率如下所示：

| ---  |  $\eta = 0.5$ | $\eta = 0.2$| $\eta = 0.1$| $\eta = 0.05$ |$\eta = 0.02$ | $\eta = 0.01$ |$\eta = 0.005$ |$\eta = 0.001$ |
| ---  | ------   | ------| -- | --   | --  |  -   |  --  | -- |
|$iter = 5$ |  0.5818   |0.8907     |0.8966     |0.8942     |0.8991     |0.8844     |0.8627     |0.7398| 
|$iter = 10$  | 0.5588  |0.8941     |0.9072     |0.8996     |0.9089     |0.9007     |0.8890     |0.8116| 
|$iter = 20$  | 0.7014  |0.8996     |0.9052     |0.9075     |0.9192     |0.9146     |0.9070     |0.8645| 
|$iter = 50$  | 0.6625  |0.9003     |0.9038     |0.9075     |0.9170     |0.9168     |0.9164     |0.8872| 
|$iter = 100$| 0.7409   |0.9015     |0.8976     |0.8986     |0.9098     |0.9262     |0.9149     |0.9124| 
|$iter = 200$| 0.8127   |0.8962     |0.8902     |0.8975     |0.9103     |0.9150     |0.9192     |0.9176| 
|$iter = 300$| 0.8109   |0.8971     |0.8919     |0.8886     |0.9034     |0.9050     |0.9170     |0.9160| 
|$iter = 500$| 0.6836   |0.9006     |0.8959     |0.8898     |0.8973     |0.9069     |0.9153     |0.9219| 

从上可以看出：
1. 学习效率为 $ 0 < \eta < 1$，$\eta$ 太大效果不好，太小收敛速度太慢
2. 并不是迭代次数越多多好，有些时候随着迭代次数的增加测试的精度反而下降
3. $\eta = 0.02, iter = 20$就以取得较好的效果

### 5. 完整代码和数据：

[我的GitHub](https://github.com/muzichao/MachineLearning/tree/master/BP%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E7%9A%84%E6%95%B0%E6%8D%AE%E5%88%86%E7%B1%BB-OCR%E6%96%87%E6%9C%AC%E8%AF%86%E5%88%AB)
### 6. 参考

1. 《机器学习》第四章-人工神经网络
2. [神经网络推导](http://blog.csdn.net/endlch/article/details/46933861)
3. 《MATLAB神经网络30个案例分析》第一个案例
