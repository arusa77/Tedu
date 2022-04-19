'''波士顿房屋价格预测'''

import paddle
import paddle.fluid as fluid
import matplotlib.pyplot as plt
import os
import numpy as np

# 数据准备
BUF_SIZE = 500
BATCH_SIZE = 20

random_reader = paddle.reader.shuffle(paddle.dataset.uci_housing.train(),
                                      buf_size=BUF_SIZE)

train_reader = paddle.batch(random_reader,batch_size=BATCH_SIZE)

# 模型搭建
x = fluid.layers.data(name='xxx',shape=[13],dtype='float32')
y = fluid.layers.data(name='yyy',shape=[1],dtype='float32')

# 定义全连接模型
pred_y = fluid.layers.fc(input=x,
                         size=1,
                         act=None)
#损失函数
cost = fluid.layers.square_error_cost(input=pred_y,
                                      label=y)
avg_cost = fluid.layers.mean(cost)

optimizer = fluid.optimizer.SGD(learning_rate=0.001)
optimizer.minimize(avg_cost)

# 模型训练，保存
place = fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())#初始化

EPOCH_NUM = 100

iters = []
train_costs = []
iter = 0

#参数喂入器
feeder = fluid.DataFeeder(feed_list=[x,y],place=place)

for pass_id in range(EPOCH_NUM):
    i = 0 # 计算总批次
    for data in train_reader():
        i += 1
        train_cost = exe.run(program=fluid.default_main_program(),
                             feed=feeder.feed(data),
                             fetch_list=[avg_cost])

        if i % 10 == 0:
            print('pass_id:{},cost:{}'.format(pass_id,train_cost[0][0]))

        iter = iter + BATCH_SIZE
        iters.append(BATCH_SIZE)
        train_costs.append(train_cost[0][0])

    # 训练可视化
    plt.figure('Train_Cost')
    plt.plot(iters,train_costs,color='orangered')
    plt.grid(linestyle=':')
    plt.savefig('train.png')

# 模型的保存
model_path = 'model/uci_housing'#模型保存路径
if not os.path.exists(model_path):
    os.makedirs(model_path)

fluid.io.save_inference_model(model_path,#模型保存路径
                              ['xxx'],#调用模型预测时，需要传入的参数
                              [pred_y],#模型预测结果从哪取
                              exe)#执行器
# 创建新的执行器，执行模型的加载，并预测
infer_exe = fluid.Executor(place=place)
#加载模型 预测
infer_program,\
feed_names,\
fetch_names = fluid.io.load_inference_model(model_path,
                                            infer_exe)

# 测试集reader

infer_reader = paddle.batch(paddle.dataset.uci_housing.test(),
                            batch_size=200)

test_data = next(infer_reader())# 获取一批数据
test_x = np.array([data[0] for data in test_data]).astype('floate32')
test_y = np.array([data[1] for data in test_data]).astype('floate32')

x_name = feed_names[0]
res = infer_exe.run(infer_program,
                    feed = {x_name:test_x},#喂入函数
                    fetch_list=fetch_names)#获取结果




