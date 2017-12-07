import dataset
import weight
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, 1024])
y = tf.placeholder(tf.float32, shape=[None, 2])

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1, 32, 32, 1])

s1 = conv2d(x_image, W_conv1) + b_conv1
h_conv1 = s1 * tf.nn.sigmoid(s1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

s2 = conv2d(h_pool1, W_conv2) + b_conv2
h_conv2 = s2 * tf.nn.sigmoid(s2)
h_pool2 = max_pool_2x2(h_conv2)

W_conv3 = weight_variable([5, 5, 64, 128])
b_conv3 = bias_variable([128])

s3 = conv2d(h_pool2, W_conv3) + b_conv3
h_conv3 = s3 * tf.nn.sigmoid(s3)
h_pool3 = max_pool_2x2(h_conv3)

W_fc1 = weight_variable([4 * 4 * 128, 1024])
b_fc1 = bias_variable([1024])

h_pool3_flat = tf.reshape(h_pool3, [-1, 4 * 4 * 128])
s4 = tf.matmul(h_pool3_flat, W_fc1) + b_fc1
h_fc1 = s4 * tf.nn.sigmoid(s4)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 2])
b_fc2 = bias_variable([2])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

diff = y - y_conv
error = tf.norm(diff, axis=1)
loss = tf.reduce_mean(error)
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

dataset_size = 30000
batch_size = 100
num_batchs = dataset_size // batch_size
num_epochs = int(input("Please enter the number of epochs : "))

train_dataset_manager = dataset.TutorialDatasetManager(dataset_size=dataset_size)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    start_epoch_i = weight.load_weights_with_confirm(saver, sess)

    for epoch_i in range(num_epochs):
        batch_x, batch_y = train_dataset_manager.next_batch(5)
        # train_loss = loss.eval(feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
        train_error = diff.eval(feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
        #print(str(start_epoch_i + epoch_i) + " epoch train_loss : " + str(train_loss))
        print(str(start_epoch_i + epoch_i) + " epoch train error : " + str(train_error))

        for batch_i in range(num_batchs):
            batch_x, batch_y = train_dataset_manager.next_batch(batch_size)
            train_step.run(feed_dict={x: batch_x, y: batch_y, keep_prob: 0.5})

        weight.save_weights(saver, sess, start_epoch_i + epoch_i)

    test_dataset_manager = dataset.TutorialDatasetManager(dataset_size=100)

    test_x = test_dataset_manager.dataset_x
    test_y = test_dataset_manager.dataset_y
    print('test loss %g' % loss.eval(feed_dict={x: test_x, y: test_y, keep_prob: 1.0}))
    
    pred_y = y_conv.eval(feed_dict={x: test_x, y: test_y, keep_prob: 1.0})
    test_error = diff.eval(feed_dict={x: test_x, y: test_y, keep_prob: 1.0})
    plt_nrow = 2
    plt_ncol = 2
    fig = plt.figure()
    for i in range(plt_nrow * plt_ncol):
        sub = fig.add_subplot(plt_nrow, plt_ncol, i + 1)
        sub.set_title(str(test_y[i]) + "\n" + str(pred_y[i]) + "\n" + str(test_error[i]))
        plt.imshow(np.reshape(test_x[i], (32, 32)), cmap="gray")
        plt.plot(pred_y[i][0], pred_y[i][1], "+r")
    plt.show()
