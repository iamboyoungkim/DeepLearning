import tensorflow as tf
import numpy as np
import random
from tensorflow.python.training import checkpoint_utils
from glob import glob
import os
from PIL import Image

Wxh = tf.Variable(tf.random_normal([3, 3, 2, 32], stddev=0.01, name='Wxh'))  # 3x3x1 conv, 64 outputs
Whh = tf.Variable(tf.random_normal([3, 3, 32, 32], stddev=0.01, name='Whh'))  # 3x3x64 conv, 64 outputs
Whh2 = tf.Variable(tf.random_normal([3, 3, 32, 32], stddev=0.01, name='Whh2'))  # 3x3x64 conv, 64 outputs
Whh3 = tf.Variable(tf.random_normal([3, 3, 32, 32], stddev=0.01, name='Whh3'))  # 3x3x64 conv, 64 outputs
Why = tf.Variable(tf.random_normal([3, 3, 32, 1], stddev=0.01, name='Why'))  # 3x3x64 conv, 1 output

params = {
    'Wxh':Wxh,
    'Whh':Whh,
    'Whh2':Whh2,
    'Whh3':Whh3,
    'Why':Why
}
saver = tf.train.Saver(params)

X = tf.placeholder("float32", [None, None, None, 1])
Y = tf.placeholder("float32", [None, None, None, 1])
H = tf.placeholder("float32", [None, None, None, 32])

Z1 = tf.nn.conv2d(tf.concat([X, X], 3), Wxh, strides=[1, 1, 1, 1], padding='SAME')
H1 = tf.nn.relu(tf.nn.conv2d(Z1, Whh, strides=[1, 1, 1, 1], padding='SAME') + tf.nn.conv2d(H, Whh2, strides=[1, 1, 1, 1], padding='SAME'))
Hu1 = tf.nn.relu(tf.nn.conv2d(H1,Whh3, strides=[1, 1, 1, 1], padding='SAME'))
Y1 = tf.nn.conv2d(Hu1, Why, strides=[1, 1, 1, 1], padding='SAME')

Z2 = tf.nn.conv2d(tf.concat([X, Y1], 3), Wxh, strides=[1, 1, 1, 1], padding='SAME')
H2 = tf.nn.relu(tf.nn.conv2d(Z2, Whh, strides=[1, 1, 1, 1], padding='SAME') + tf.nn.conv2d(H1, Whh2, strides=[1, 1, 1, 1], padding='SAME'))
Hu2 = tf.nn.relu(tf.nn.conv2d(H2,Whh3, strides=[1, 1, 1, 1], padding='SAME'))
Y2 = tf.nn.conv2d(Hu2, Why, strides=[1, 1, 1, 1], padding='SAME')

Z3 = tf.nn.conv2d(tf.concat([X, Y2], 3), Wxh, strides=[1, 1, 1, 1], padding='SAME')
H3 = tf.nn.relu(tf.nn.conv2d(Z3, Whh, strides=[1, 1, 1, 1], padding='SAME') + tf.nn.conv2d(H2, Whh2, strides=[1, 1, 1, 1], padding='SAME'))
Hu3 = tf.nn.relu(tf.nn.conv2d(H3,Whh3, strides=[1, 1, 1, 1], padding='SAME'))
Y3 = tf.nn.conv2d(Hu3, Why, strides=[1, 1, 1, 1], padding='SAME')

psnr = tf.reduce_mean(tf.image.psnr(Y3, Y, max_val=255))

print("validation...")
img_list = glob('./SR_dataset/Set5/*.png')
    
for i in range(len(img_list)):
    img = Image.open(img_list[i])
    (img_h, img_w) = img.size
    
    with tf.Session() as sess:
        img_gray = tf.image.rgb_to_grayscale(img)
        img = tf.cast(img_gray, dtype=tf.float32)
        img = tf.reshape(img, [1, img_w, img_h, 1]).eval()

        new_img = tf.image.resize_images(img, ((int)(img_w / 2), (int)(img_h / 2)))
        renew_img = tf.image.resize_images(new_img, (img_w, img_h)).eval()
        H_validation = np.zeros([1, img_w, img_h, 32])
        
        saver.restore(sess, "./checkpoint/RNN_model-5000")
        
        test, psnr_ = sess.run([Y3, psnr], feed_dict={X: renew_img, Y: img, H: H_validation})
        
        print("Validation epoch : ",i + 1, " Psnr : ", psnr_)
        save_img = tf.cast(test,tf.uint8)
        save_img = save_img.eval()
        save_img = np.reshape(save_img, [img_w, img_h])

        save = Image.fromarray(save_img)
        new_file = "{}.png".format("{0:05d}".format(i+1))
        save.save('./SR_dataset/result/' + new_file)