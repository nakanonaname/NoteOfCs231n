# 第一份作业的问题和回答：
## KNN
**Inline Question 1**

Notice the structured patterns in the distance matrix, where some rows or columns are visibly brighter. (Note that with the default color scheme black indicates low distances while white indicates high distances.)

- What in the data is the cause behind the distinctly bright rows?
- What causes the columns?

$\color{blue}{\textit Your Answer:}$ rows：测试样本与大多数训练样本的距离都很远，说明这些测试样本可能是异常值，或者它们属于训练集中样本较少的类别，因此在训练集中找不到接近的邻居。 
columns：训练样本与所有测试样本的距离都很远，说明这些训练样本可能是噪声样本或异常点，与其他数据分布差异较大，不像任何一个测试样本。


**Inline Question 2**

We can also use other distance metrics such as L1 distance.
For pixel values $p_{ij}^{(k)}$ at location $(i,j)$ of some image $I_k$,

the mean $\mu$ across all pixels over all images is $$\mu=\frac{1}{nhw}\sum_{k=1}^n\sum_{i=1}^{h}\sum_{j=1}^{w}p_{ij}^{(k)}$$
And the pixel-wise mean $\mu_{ij}$ across all images is
$$\mu_{ij}=\frac{1}{n}\sum_{k=1}^np_{ij}^{(k)}.$$
The general standard deviation $\sigma$ and pixel-wise standard deviation $\sigma_{ij}$ is defined similarly.

Which of the following preprocessing steps will not change the performance of a Nearest Neighbor classifier that uses L1 distance? Select all that apply. To clarify, both training and test examples are preprocessed in the same way.

1. Subtracting the mean $\mu$ ($\tilde{p}_{ij}^{(k)}=p_{ij}^{(k)}-\mu$.)
2. Subtracting the per pixel mean $\mu_{ij}$  ($\tilde{p}_{ij}^{(k)}=p_{ij}^{(k)}-\mu_{ij}$.)
3. Subtracting the mean $\mu$ and dividing by the standard deviation $\sigma$.
4. Subtracting the pixel-wise mean $\mu_{ij}$ and dividing by the pixel-wise standard deviation $\sigma_{ij}$.
5. Rotating the coordinate axes of the data, which means rotating all the images by the same angle. Empty regions in the image caused by rotation are padded with a same pixel value and no interpolation is performed.

$\color{blue}{\textit Your Answer:}$ 1,2,3


$\color{blue}{\textit Your Explanation:}$ 1：所有特征均匀移位→L1距离保持不变→神经网络性能保持不变。2：独立地移动每个像素，但L1距离对每个维度的移动是不变的→对性能没有影响。3：所有特征均匀缩放→L1相对距离保持→NN预测不变。


**Inline Question 3**

Which of the following statements about $k$-Nearest Neighbor ($k$-NN) are true in a classification setting, and for all $k$? Select all that apply.
1. The decision boundary of the k-NN classifier is linear.
2. The training error of a 1-NN will always be lower than or equal to that of 5-NN.
3. The test error of a 1-NN will always be lower than that of a 5-NN.
4. The time needed to classify a test example with the k-NN classifier grows with the size of the training set.
5. None of the above.

$\color{blue}{\textit Your Answer:}$ 2&4


$\color{blue}{\textit Your Explanation:}$ 2:1-NN的训练误差总是小于或等于5-NN的。4：使用k-NN分类器对测试样例进行分类所需的时间随着训练集的大小而增长。

## FullyConnectedNets

## Inline Question 1:
Did you notice anything about the comparative difficulty of training the three-layer network vs. training the five-layer network? In particular, based on your experience, which network seemed more sensitive to the initialization scale? Why do you think that is the case?

## Answer:
五层网络：使用Xavier/Glorot初始化（按1/√n缩放）比固定缩放效果更好,批处理规范化有助于降低初始化的敏感性,更小的学习率更需要超参数调优。

这与已知的深度学习理论一致，其中更深的架构需要更仔细的初始化，以保持网络中稳定的梯度流。


## Inline Question 2:

AdaGrad, like Adam, is a per-parameter optimization method that uses the following update rule:

```
cache += dw**2
w += - learning_rate * dw / (np.sqrt(cache) + eps)
```

John notices that when he was training a network with AdaGrad that the updates became very small, and that his network was learning slowly. Using your knowledge of the AdaGrad update rule, why do you think the updates would become very small? Would Adam have the same issue?


## Answer:
AdaGrad更新变小的原因：
AdaGrad持续累加梯度平方（cache += dw**2），导致分母√cache随时间单调增大，使得学习率不断衰减，最终更新量趋近于零。

Adam是否会有同样问题？
不会。Adam使用指数衰减的梯度平方滑动平均（而非累加），并引入偏差校正，使得分母√v_hat能稳定在合理范围，避免更新量无限缩小。

## 2 layered net
## Inline Question 1:

We've only asked you to implement ReLU, but there are a number of different activation functions that one could use in neural networks, each with its pros and cons. In particular, an issue commonly seen with activation functions is getting zero (or close to zero) gradient flow during backpropagation. Which of the following activation functions have this problem? If you consider these functions in the one dimensional case, what types of input would lead to this behaviour?
1. Sigmoid
2. ReLU
3. Leaky ReLU

$\color{blue}{\textit Your Answer:}$ *Fill this in*
## Inline Question 1:

We've only asked you to implement ReLU, but there are a number of different activation functions that one could use in neural networks, each with its pros and cons. In particular, an issue commonly seen with activation functions is getting zero (or close to zero) gradient flow during backpropagation. Which of the following activation functions have this problem? If you consider these functions in the one dimensional case, what types of input would lead to this behaviour?
1. Sigmoid
2. ReLU
3. Leaky ReLU

$\color{blue}{\textit Your Answer:}$ 会出现梯度消失问题的激活函数：

Sigmoid：导数σ'(x)=σ(x)(1-σ(x))，在输入值较大时σ(x)接近0或1，导致导数接近0

ReLU：导数在负区间恒为0，导致"神经元死亡"现象

不会出现梯度消失问题的激活函数：

Leaky ReLU：全区间都有非零梯度
Leaky ReLU通过负区间的非零斜率设计，从根本上避免了梯度消失问题。

## Inline Question 2:

Now that you have trained a Neural Network classifier, you may find that your testing accuracy is much lower than the training accuracy. In what ways can we decrease this gap? Select all that apply.

1. Train on a larger dataset.
2. Add more hidden units.
3. Increase the regularization strength.
4. None of the above.

$\color{blue}{\textit Your Answer:}$ 1&3

$\color{blue}{\textit Your Explanation:}$ 1. 使用更大的训练集：更多数据能减少过拟合，使模型学到更通用的特征而非训练集噪声。3：增强正则化强度：更强的L2/L1正则化或Dropout能约束模型复杂度，抑制过拟合。
2：增加隐藏单元：会增大模型容量，反而可能扩大训练-测试差距（除非同时调整正则化）。

## softmax

**Inline Question 1**

Why do we expect our loss to be close to -log(0.1)? Explain briefly.**

$\color{blue}{\textit Your Answer:}$ Softmax归一化后，每个类别的预测概率≈1/C（C为类别数）

10分类时：-log(1/10) = log(10) ≈ 2.302

实际训练时初始损失若接近该值，说明初始化正确

若显著偏离（如远大于该值），可能提示初始化或实现有问题

p_i = 1/C  
L = -log(p_i) = -log(1/C) = log(C)




