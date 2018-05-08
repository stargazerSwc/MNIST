import tensorflow as tf

# 定义x用来存储图像，28*28=784，None标识可以用来存储无限个784变量
x = tf.placeholder(tf.float32, [None, 784])

'''
    定义softmax模型中的权值与偏置，此模型可认为输入为28*28图片一维化后的向量
    中间层为10个神经元，然后将每个神经元的输出输入到softmax模型里，获得规范化后的概率，进行分类
'''
W = tf.Variable(tf.zeros[784, 10])
b = tf.Variable(tf.zeros[10])

# 构建输出
y = tf.nn.softmax(tf.matmul(x, W) + b)

'''
    构建Loss Function
    Loss Function是交叉熵
    注意y_是None张图像的分类标签结果
    y是当前神经网络的分类结果
'''
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entroy = -tf.reduce_sum(y_ * tf.log(y))

