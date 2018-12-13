import tensorflow as tf

w_init1 = tf.random_normal_initializer(stddev=0.02)
# w_init2 = tf.Variable(tf.random_normal([1], stddev=1, seed=1), name='222')
w = tf.Variable(tf.constant())
w_init = tf.glorot_uniform_initializer()
# w = [44]
add = w + 1

print(w_init)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # sess.run(w_init)

    print(sess.run(tf.report_uninitialized_variables()))
    # print(sess.run(w_init))
    print(sess.run([add], {w=44}))

