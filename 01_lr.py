'''简单的线性回归，熟悉paddle处理模型'''

import paddle.fluid as fluid
import numpy as np
import matplotlib.pyplot as plt

# 定义样本
train_data = np.array([[0.5],[0.6],[0.8],[1.1],[1.4]]).astype('float32')
y_true = np.array([[5.0],[5.5],[6.0],[6.8],[7.1]]).astype('float32')

# 定义变量
x = fluid.layers.data(name='xxx',shape=[1],dtype='float32')
y = fluid.layers.data(name='yyy',shape=[1],dtype='float32')

# 定义操作
# 构建全连接模型
pred_y = fluid.layers.fc(input=x,# 输入数据
                            size=1,#输出值个数
                            act=None)# 激活函数

# 损失函数：均方误差
# 没平均的结果
cost = fluid.layers.square_error_cost(input=pred_y,#预测值
                                      label=y)# 真实值

avg_cost = fluid.layers.mean(cost)# 均方误差

# 随机梯度下降优化器
optimizer = fluid.optimizer.SGD(learning_rate=0.01)
optimizer.minimize(avg_cost)

#执行器
place = fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())

iters = []
costs = []








