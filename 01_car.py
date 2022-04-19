'''变量'''

import paddle.fluid as fluid
import numpy as np

x = fluid.layers.data(name='xxx',shape=[2,3],dtype='float32')
y = fluid.layers.data(name='yyy',shape=[2,3],dtype='float32')

x_add_y = fluid.layers.elementwise_add(x,y)
x_mull_y = fluid.layers.elementwise_mul(x,y)

# 执行
place = fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())

val1 = np.arange(1,7).reshape(2,3)
val2 = np.zeros([2,3])

params = {x:val1,
          y:val2}

outs = exe.run(program=fluid.default_main_program(),
                feed=params,
                fetch_list=[x_add_y,x_mull_y])

for i in outs:
    print(i)