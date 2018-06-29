import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', validation_size=0)

# shape of images: [number_of_images, 28, 28, 1]
def apply_noise(images):
    noise = noise_factor * np.random.randn(np.shape(images)[0], 28, 28, 1)
    return np.clip(images + noise, 0, 1)


# hyperparameters
learning_rate = 0.001
epochs = 0
batch_size = 128
load_model = True
save_every = 10
noise_factor = 0.3

# model
inputs = tf.placeholder(shape=[None, 28, 28, 1], dtype=tf.float32)
targets = tf.placeholder(shape=[None, 28, 28, 1], dtype=tf.float32)

# encoder
conv1 = tf.layers.conv2d(inputs, filters=16, kernel_size=(3, 3), padding='same', activation=tf.nn.relu) # -> 28x28x16
maxpool1 = tf.layers.max_pooling2d(conv1, pool_size=(2, 2), strides=(2, 2), padding='same') # -> 14x14x16
conv2 = tf.layers.conv2d(maxpool1, filters=8, kernel_size=(3, 3), padding='same', activation=tf.nn.relu) # 14x14x8
maxpool2 = tf.layers.max_pooling2d(conv2, pool_size=(2, 2), strides=(2, 2), padding='same') # 7x7x8
conv3 = tf.layers.conv2d(maxpool2, filters=8, kernel_size=(3, 3), padding='same', activation=tf.nn.relu) # -> 7x7x8
encoded = tf.layers.max_pooling2d(conv3, pool_size=(2, 2), strides=(2, 2), padding='same') # ->4x4x8

# decoder
upsample1 = tf.image.resize_images(encoded, size=(7, 7), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR) # -> 7x7x8
conv4 = tf.layers.conv2d(upsample1, filters=8, kernel_size=(3, 3), padding='same', activation=tf.nn.relu) # 7x7x8
upsample2 = tf.image.resize_images(conv4, (14, 14), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR) # -> 14x14x8
conv5 = tf.layers.conv2d(upsample2, filters=8, kernel_size=(3, 3), padding='same', activation=tf.nn.relu) # -> 14x14x8
upsample3 = tf.image.resize_images(conv5, (28, 28), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR) # -> 28x28x8
conv6 = tf.layers.conv2d(upsample3, filters=16, kernel_size=(3, 3), padding='same', activation=tf.nn.relu) # -> 28x28x16

logits = tf.layers.conv2d(conv6, filters=1, kernel_size=(3, 3), padding='same', activation=None)
output = tf.nn.sigmoid(logits)
loss = tf.reduce_sum(tf.square(logits - targets))
trainer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

saver = tf.train.Saver()
with tf.Session() as sess:
    if load_model:
        print('restoring model...', end='')
        saver.restore(sess, './model.ckpt')
        print('done')
    else:
        sess.run(tf.global_variables_initializer())

    for i in range(epochs):
        print('Epoch', i, 'running...', end='')
        epoch_loss = 0

        for ii in range(mnist.train.num_examples//batch_size):
            batch = mnist.train.next_batch(batch_size)[0].reshape((-1, 28, 28, 1))
            l, _ = sess.run([loss, trainer], feed_dict={inputs: apply_noise(batch), targets: batch})
            epoch_loss += l

        print('done; total loss: ', epoch_loss)

        if (i + 1) % save_every == 0:
            print('saving model...', end='')
            saver.save(sess, './model.ckpt')
            print('done')

    fig, axes = plt.subplots(nrows=2, ncols=10, sharex=True, sharey=True, figsize=(20, 4))
    in_imgs = apply_noise(mnist.test.images[:10].reshape((10, 28, 28, 1)))
    reconstructed = sess.run(output, feed_dict={inputs: in_imgs})

    for images, row in zip([in_imgs, reconstructed], axes):
        for img, ax in zip(images, row):
            ax.imshow(img.reshape((28, 28)), cmap='Greys_r')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

    fig.tight_layout(pad=0.1)
    plt.show()