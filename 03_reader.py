'''数据读取器自定义'''

import paddle

# 原始读取器

def reader_creator(file_path):
    def reader():
        with open(file_path,'r') as f:
            lines = f.readlines()
            for line in lines:
                # yield line[:-1]
                yield line.replace('\n','')

    return reader # 不能写函数调用，要写函数名

reader = reader_creator('./test.txt')
# 对reader 进行包装，包装称随机读取器
shuffle_reader = paddle.reader.shuffle(reader,10)

# 对reader 批量进行，包装称随机读取器
batch_reader = paddle.batch(shuffle_reader,3)

for data in batch_reader():
    print(data)

# for data in reader():
#     print(data)
