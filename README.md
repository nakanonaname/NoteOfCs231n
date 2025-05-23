# 第一份作业的问题和回答：
**Inline Question 1**

Notice the structured patterns in the distance matrix, where some rows or columns are visibly brighter. (Note that with the default color scheme black indicates low distances while white indicates high distances.)

- What in the data is the cause behind the distinctly bright rows?
- What causes the columns?

$\color{blue}{\textit Your Answer:}$ rows：测试样本与大多数训练样本的距离都很远，说明这些测试样本可能是异常值，或者它们属于训练集中样本较少的类别，因此在训练集中找不到接近的邻居。 columns：训练样本与所有测试样本的距离都很远，说明这些训练样本可能是噪声样本或异常点，与其他数据分布差异较大，不像任何一个测试样本。
