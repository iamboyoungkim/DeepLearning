import tensorflow as tf
from PIL import Image
import numpy as np
import random
from glob import glob
import os

# -*- coding: utf-8 -*-
# refered to https://gist.github.com/vinhkhuc/7ec5bf797308279dc587
class RNN(object):
	def __init__(self, input, input2):
		self.input = input
		self.input2 = input2
		self.weights = {'Wxh': tf.Variable(tf.random_normal([3, 3, 2, 32], stddev=0.01)),
						'Whh': tf.Variable(tf.random_normal([3, 3, 32, 32], stddev=0.01)),
						'Whh2': tf.Variable(tf.random_normal([3, 3, 32, 32], stddev=0.01)),
						'Whh3': tf.Variable(tf.random_normal([3, 3, 32, 32], stddev=0.01)),
						'Why': tf.Variable(tf.random_normal([3, 3, 32, 1], stddev=0.01))
						}
		self.output = self.build_model(self.input, self.input2)
	
	def build_model(self, X, H):
		param_list = {
			'Wxh':self.weights['Wxh'],
			'Whh':self.weights['Whh'],
			'Whh2':self.weights['Whh2'],
			'Whh3':self.weights['Whh3'],
			'Why':self.weights['Why']
		}
		Z1 = tf.nn.conv2d(tf.concat([X, X], 3), self.weights['Wxh'], strides=[1,1,1,1], padding='SAME')
		H1 = tf.nn.relu(tf.nn.conv2d(Z1, self.weights['Whh'], strides=[1,1,1,1], padding="SAME") + tf.nn.conv2d(H, self.weights['Whh2'], strides=[1,1,1,1], padding="SAME"))
		Hu1 = tf.nn.relu(tf.nn.conv2d(H1, self.weights['Whh3'], strides=[1,1,1,1], padding="SAME"))
		Y1 = tf.nn.conv2d(Hu1, self.weights['Why'], strides=[1,1,1,1], padding="SAME")

		Z2 = tf.nn.conv2d(tf.concat([X, Y1], 3), self.weights['Wxh'], strides=[1,1,1,1], padding='SAME')
		H2 = tf.nn.relu(tf.nn.conv2d(Z2, self.weights['Whh'], strides=[1,1,1,1], padding="SAME") + tf.nn.conv2d(H1, self.weights['Whh2'], strides=[1,1,1,1], padding="SAME"))
		Hu2 = tf.nn.relu(tf.nn.conv2d(H2, self.weights['Whh3'], strides=[1,1,1,1], padding="SAME"))
		Y2 = tf.nn.conv2d(Hu2, self.weights['Why'], strides=[1,1,1,1], padding="SAME")

		Z3 = tf.nn.conv2d(tf.concat([X, Y2], 3), self.weights['Wxh'], strides=[1,1,1,1], padding='SAME')
		H3 = tf.nn.relu(tf.nn.conv2d(Z3, self.weights['Whh'], strides=[1,1,1,1], padding="SAME") + tf.nn.conv2d(H2, self.weights['Whh2'], strides=[1,1,1,1], padding="SAME"))
		Hu3 = tf.nn.relu(tf.nn.conv2d(H3, self.weights['Whh3'], strides=[1,1,1,1], padding="SAME"))
		Y3 = tf.nn.conv2d(Hu3, self.weights['Why'], strides=[1,1,1,1], padding="SAME")

		return Y1, Y2, Y3, param_list

def data_crop(input_path):
	tmp = random.randrange(len(input_path))
	img = Image.open(input_path[tmp])
	(img_h,img_w) = img.size

	range_h = (int)(img_h / 32)
	range_w = (int)(img_w / 32)

	random_h = random.randrange(range_h)
	random_w = random.randrange(range_w)

	bbox = (random_h * 32, random_w * 32, (random_h + 1) * 32, (random_w + 1) * 32)
	crop_img = img.crop(bbox)

	return crop_img

def data_resize(crop_img):
	tmp = tf.image.resize_images(crop_img,(32,32))
	img_gray = tf.image.rgb_to_grayscale(tmp)
	img = tf.cast(img_gray, dtype=tf.float32)
	img = tf.reshape(img,[1,32,32,1])

	new_img = tf.image.resize_images(img,(16,16))
	renew_img = tf.image.resize_images(new_img,(32,32))

	return img, renew_img
	
# def load(sess, checkpoint_dir, checkpoint_num, saver):
# 	print(" [*] Reading checkpoints...")
# 	ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
# 	if ckpt and ckpt.model_checkpoint_path:
# 		ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
# 		ckpt_idx = ckpt_name.find("-")
# 		ckpt_name = ckpt_name[:ckpt_idx+1] + str(checkpoint_num)
# 		saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
# 		return int(checkpoint_num)
# 	else: return -1

# def save(sess, checkpoint_dir, step, saver):
# 	model_name = "RNN.model"
# 	if not os.path.exists(checkpoint_dir):
# 		os.makedirs(checkpoint_dir)
# 	saver.save(sess, os.path.join(checkpoint_dir, model_name), global_step=step)

ckpt_dir = "checkpoint"
train_summary = "log/train_log"
test_summary = "log/test_log"

def get_result(input_path, config):
	X = tf.placeholder("float32", [None, None, None, 1])
	Y = tf.placeholder("float32", [None, None, None, 1])
	H = tf.placeholder("float32", [None, None, None, 32])
	C = tf.placeholder("float32", [None, None, None])

	epoch_num = 5000
	mini_batch = 128
	ckpt_dir = "checkpoint"
	train_summary_dir = "log/train_log"
	test_summary_dir = "log/test_log"
	eval_path = "/Users/kimboyoung/Desktop/4-1/deep_learning/2019_ITE4053_2015004402/lab4/SR_dataset/Set5"
	result_path = 'result'

	model = RNN(X, H)
	(hypo1, hypo2, hypo3, param_list) = model.output
	loss = tf.reduce_mean(tf.square(hypo1-Y) + tf.square(hypo2-Y) + tf.square(hypo3-Y))
	cost_sum = tf.summary.scalar("cost", loss)

	optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
	psnr = tf.reduce_mean(tf.image.psnr(hypo3, Y, max_val=255))
	psnr_sum = tf.summary.scalar("psnr", psnr)
	all_summary = tf.summary.merge_all()

	# make new session
	sess = tf.Session()
	saver = tf.train.Saver(param_list)
	sess.run(tf.global_variables_initializer())

	cropping, resizing = data_resize(C)

	train_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)
	test_writer = tf.summary.FileWriter(test_summary_dir, sess.graph)

	# if config != -1: ckpt = load(sess, ckpt_dir, config, saver)
	# if ckpt != -1:
	# 	print("loading checkpoint.")
	# else:
	# 	print("fail in loading checkpoint.")
	# 	ckpt = 0

	for epoch in range(epoch_num):
		mini_batch = 128
		test_batch = 32

		# training
		for i in range(mini_batch):
			ori_img, resize_img = sess.run([cropping, resizing], feed_dict={C: data_crop(input_path)})
			if i == 0:
				ori = ori_img
				res = resize_img
			else:
				ori = np.append(ori, ori_img, axis=0)
				res = np.append(res, resize_img, axis=0)
			H_train = np.zeros([128,32,32,32])
		s, _, cost = sess.run([all_summary, optimizer, loss], feed_dict={X: res, Y: ori, H: H_train})
		train_writer.add_summary(s, global_step=epoch + 1)
		print("Train Epoch: ", epoch + 1, ", Cost: ", cost)

		# test
		for i in range(test_batch):
			ori_img, resize_img = sess.run([cropping, resizing], feed_dict={C: data_crop(input_path)})
			if i == 0:
				test_ori = ori_img
				test_res = resize_img
			else:
				test_ori = np.append(ori, ori_img, axis=0)
				test_res = np.append(res, resize_img, axis=0)
			H_test = np.zeros([129,32,32,32])
		s, psnr_ = sess.run([psnr_sum, psnr], feed_dict={X: test_res, Y: test_ori, H: H_test})
		test_writer.add_summary(s, global_step=epoch + 1)
		print("Psnr: ", psnr_)

		if (epoch + 1) % 1000 == 0:
			model_name = "RNN_model"
			if not os.path.exists("checkpoint"):
				os.makedirs("checkpoint")
			saver.save(sess, os.path.join("checkpoint", model_name), global_step=epoch + 1)

# get result
input_path = glob("/Users/kimboyoung/Desktop/4-1/deep_learning/2019_ITE4053_2015004402/lab4/SR_dataset/291/*.bmp")
input_path += glob("/Users/kimboyoung/Desktop/4-1/deep_learning/2019_ITE4053_2015004402/lab4/SR_dataset/291/*.jpg")
config_path = "/Users/kimboyoung/Desktop/4-1/deep_learning/2019_ITE4053_2015004402/lab4"
get_result(input_path, config_path)