# import tensorflow as tf
# import numpy as np
# from tensorlayer.layers import *
#
#
# def GAN_g1(t_image, reuse=False):
#     w_init = tf.random_normal_initializer(stddev=0.02)
#     with tf.variable_scope("GAN_g1", reuse=reuse) as vs:
#         n = InputLayer(t_image, name='in')
#         n = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='n64s1/c')
#         return n
#
#
#
# x_data = np.random.rand(100).astype(np.float32)     ##输入值[0，1)之间的随机数
# print(x_data)
# y_data = x_data * 0.1 + 0.3     ##预测值
#
# ###creat tensorflow structure strat###
# # 构造要拟合的线性模型
# Weights = tf.Variable(tf.random_uniform([1],-1.0,1.0))
# biases = tf.Variable(tf.zeros([1]))
# y = Weights * x_data + biases
#
# # 定义损失函数和训练方法
# loss = tf.reduce_mean(tf.square(y-y_data))   ##最小化方差
# optimizer = tf.train.GradientDescentOptimizer(0.5)
# train = optimizer.minimize(loss)
#
# # 初始化变量
# init = tf.initialize_all_variables()
# ###creat tensorflow structure end###
#
# # 启动
# sess = tf.Session()
# sess.run(init)
#
# # 训练拟合，每一步训练队Weights和biases进行更新
# for step in range(201):
#     sess.run(train)
#     if step % 20 == 0:
#             print(step,sess.run(Weights),sess.run(biases))
#
#
#
# import tensorflow as tf
# from tensorlayer.layers import *
# from PIL import Image
# import tensorlayer as tl
# import numpy as np
# import panda2
#
# images = Image.open("./dog.jpg")
# images = tf.image.resize_images(images, size=[416, 416], method=0,
#                                            align_corners=False)
# images = np.array(tf.reshape(images, shape=[1, 416, 416, 3]))
# print(images)
# # images = np.expand_dims(images, axis = 0)
# # print(images)
# imagess = tf.placeholder('float32', [1, 416, 416, 3], name='img')
# nn = panda2.GAN_g1(imagess)
# loss = tl.cost.sigmoid_cross_entropy(nn, tf.ones_like(nn), name='d1')
# train_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='GAN_g1')
# g_optim = tf.train.AdamOptimizer(1e-4, beta1=0.9).minimize(loss, var_list=train_var)
#
# init = tf.global_variables_initializer()
# with tf.Session() as sess:
#     # sess.run(w_init)
#
#     sess.run([loss, g_optim], feed_dict={imagess: images})
#     # print("loss = {}")
#     print("loss %s" % (loss))

import tensorflow as tf
#
# t = tf.truncated_normal_initializer(stddev=0.1, seed=1)
# v = tf.get_variable('v', [1], initializer=t)
#
# with tf.Session() as sess:
#     for i in range(1, 10, 1):
#         sess.run(tf.global_variables_initializer())
#         print(sess.run(v))


# with tf.variable_scope(tf.get_variable_scope(), reuse=False):
assert tf.get_variable_scope().reuse == False
# print(tf.get_variable_scope().name)
tf.get_variable_scope().reuse_variables()
assert tf.get_variable_scope().reuse == True

with tf.variable_scope("foo"):
        # Opened a sub-scope, still not reusing.
    print(tf.get_variable_scope().name)
    assert tf.get_variable_scope().reuse == True
    with tf.variable_scope("aa"):
        print(tf.get_variable_scope().name)
        # print(tf.get_variable_scope().name)
with tf.variable_scope("foo", reuse=False):
        # Explicitly opened a reusing scope.
    assert tf.get_variable_scope().reuse == False
    with tf.variable_scope("bar"):
            # Now sub-scope inherits the reuse flag.
        assert tf.get_variable_scope().reuse == True
    # Exited the reusing scope, back to a non-reusing one.
assert tf.get_variable_scope().reuse == False




