# MSRA_neural_network
# Multi_Variable_Linear_Regression

Code based on `https://github.com/microsoft/ai-edu/tree/master/B-%E6%95%99%E5%AD%A6%E6%A1%88%E4%BE%8B%E4%B8%8E%E5%AE%9E%E8%B7%B5/B6-%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E5%9F%BA%E6%9C%AC%E5%8E%9F%E7%90%86%E7%AE%80%E6%98%8E%E6%95%99%E7%A8%8B/SourceCode/ch05-MultiVariableLinearRegression`
（感谢老师的教导）

- 在设置超参时，先试探性将学习率调至0.001，将最大迭代次数适当调高至2000，batchsize设置为5，eps保持1e-5
  经过多次调整最终发现最佳学习率为0.01

- 其中，原程序中
```
batch_Y = self.YTrain[start:end,:]
```
处，在处理Boston房价数据集时会出现`TMI`的错误，原因是此时处理完的Y并非一维矩阵
为解决该错误，在`HelperClass.DataReader_1_2.NomalizeY`中，在处理数据时直接将Y转为一维矩阵，然后在`GetBatchTrainSamples`中将代码改为
```
batch_Y = self.YTrain[start:end]
```
问题解决
