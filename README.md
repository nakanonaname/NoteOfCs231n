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


