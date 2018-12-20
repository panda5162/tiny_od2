import tensorflow as tf
from tensorlayer.layers import *
from PIL import Image
import tensorlayer as tl
import numpy as np


def GAN_g1(t_image, reuse=False):
    w_init = tf.random_normal_initializer(stddev=0.02)


    with tf.variable_scope("GAN_g1", reuse=reuse) as vs:
        n = InputLayer(t_image, name='in')

        n = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='n64s1/c')
        return n

