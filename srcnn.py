import tensorflow as tf
import sys, os
import numpy as np
from PIL import Image

# -*- coding: utf-8 -*-
# refered to https://github.com/tegg89/SRCNN-Tensorflow
class SRCNN(object):
	def __init__(self, input):
		self.input = input
		self.weights = {'W1': tf.Variable(tf.random_normal([3, 3, 1, 64], stddev=0.01)),
						'W2': tf.Variable(tf.random_normal([3, 3, 64, 64], stddev=0.01)),
						'W3': tf.Variable(tf.random_normal([3, 3, 64, 1], stddev=0.01))}

		self.bias = {'B1': tf.Variable(tf.zeros([64])),
					 'B2': tf.Variable(tf.zeros([64])),
					 'B3': tf.Variable(tf.zeros([1]))}
		self.output = self.build_model(self.input)

	def build_model(self, X):
		L1 = tf.nn.relu(tf.nn.conv2d(X, self.weights['W1'],  # l1 shape=(?, 32, 32, 64)
						 			strides=[1, 1, 1, 1], padding="SAME") + self.bias['B1'])
		L2 = tf.nn.relu(tf.nn.conv2d(L1, self.weights['W2'],  # l2 shape=(?, 32, 32, 64)
						 			strides=[1, 1, 1, 1], padding="SAME") + self.bias['B2'])

		output = tf.nn.conv2d(L2, self.weights['W3'],  # l3 shape=(?, 32, 32, 1)
					  			strides=[1, 1, 1, 1], padding="SAME") + self.bias['B3']
		return output

def preprocessing(input_path1, input_path2):
	sess = tf.InteractiveSession()
	train_img =[]
	ori_img =[]
	img_list1 = np.array(os.listdir(input_path1))
	img_list2 = np.array(os.listdir(input_path2))

	for i in range(len(img_list1)):
		img_path = os.path.join(input_path1,img_list1[i])
		cur = Image.open(img_path)
		cur = tf.image.rgb_to_grayscale(cur).eval()
		x = np.random.randint(32, np.shape(cur)[0])
		y = np.random.randint(32, np.shape(cur)[1])
		cropped = cur[x-32:x, y-32:y]
		ori_img.append(cropped.copy())
		resize_small = tf.image.resize_images(cropped, (16, 16), tf.image.ResizeMethod.BICUBIC)
		resized_big = tf.image.resize_images(resize_small, (32, 32), tf.image.ResizeMethod.BICUBIC).eval()
		train_img.append(resized_big.copy())
	
	for i in range(len(img_list2)):
		img_path = os.path.join(input_path2,img_list2[i])
		cur = Image.open(img_path)
		cur = tf.image.rgb_to_grayscale(cur).eval()
		x = np.random.randint(32, np.shape(cur)[0])
		y = np.random.randint(32, np.shape(cur)[1])
		cropped = cur[x-32:x, y-32:y]
		ori_img.append(cropped.copy())
		resize_small = tf.image.resize_images(cropped, (16, 16), tf.image.ResizeMethod.BICUBIC)
		resized_big = tf.image.resize_images(resize_small, (32, 32), tf.image.ResizeMethod.BICUBIC).eval()
		train_img.append(resized_big.copy())
	sess.close()
	return np.array(train_img), np.array(ori_img)

def load(sess, checkpoint_dir, checkpoint_num, saver):
	print(" [*] Reading checkpoints...")
	ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
	if ckpt and ckpt.model_checkpoint_path:
		ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
		ckpt_idx = ckpt_name.find("-")
		ckpt_name = ckpt_name[:ckpt_idx+1] + str(checkpoint_num)
		saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
		return int(checkpoint_num)
	else: return -1

def save(sess, checkpoint_dir, step, saver):
	model_name = "SRCNN.model"
	if not os.path.exists(checkpoint_dir):
		os.makedirs(checkpoint_dir)
	saver.save(sess, os.path.join(checkpoint_dir, model_name), global_step=step)

ckpt_dir = "checkpoint"
train_summary = "log/train_log"
test_summary = "log/test_log"

def get_result(input_path1, input_path2, config):
	X = tf.placeholder("float32", [None, None, None, 1])
	Y = tf.placeholder("float32", [None, None, None, 1])

	epoch_num = 10000
	mini_batch = 128
	ckpt_dir = "checkpoint"
	train_summary_dir = "log/train_log"
	test_summary_dir = "log/test_log"
	eval_path = "/Users/kimboyoung/Desktop/4-1/deep_learning/2019_ITE4053_2015004402/lab2/SR_dataset/Set5"
	result_path = 'result'

	train_img, label_img = preprocessing(input_path1, input_path2)
	train_img = np.reshape(train_img, [train_img.shape[0], train_img.shape[1], train_img.shape[2], 1])
	label_img = np.reshape(label_img, [label_img.shape[0], label_img.shape[1], label_img.shape[2], 1])

	test_img = train_img[int(0.9*len(train_img)):]
	test_ori = label_img[int(0.9 * len(label_img)):]
	train_img = train_img[:int(0.9*len(train_img))]
	label_img = label_img[:int(0.9 * len(label_img))]

	model = SRCNN(input=X)
	hypothesis = model.output
	loss = tf.reduce_mean(tf.square(Y - hypothesis))
	cost_sum = tf.summary.scalar("cost", loss)

	optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
	psnr = tf.reduce_mean(tf.image.psnr(hypothesis, Y, max_val=255))
	psnr_sum = tf.summary.scalar("psnr", psnr)
	all_summary = tf.summary.merge_all()

	# make new session
	sess = tf.Session()
	saver = tf.train.Saver()
	sess.run(tf.global_variables_initializer())

	train_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)
	test_writer = tf.summary.FileWriter(test_summary_dir, sess.graph)

	if config != -1: ckpt = load(sess, ckpt_dir, config, saver)
	if ckpt != -1:
		print("loading checkpoint.")
	else:
		print("fail in loading checkpoint.")
		ckpt = 0

	for epoch in range(ckpt, epoch_num):
		batch_size = int(len(train_img) / mini_batch) + 1
		loss_all = 0

		# training
		for i in range(batch_size):
			if i == batch_size - 1:
				batch = train_img[i*mini_batch:]
				batch_ori = label_img[i*mini_batch:]
			else:
				batch = train_img[i*mini_batch:(i+1)*mini_batch]
				batch_ori = label_img[i*mini_batch:(i+1)*mini_batch]

			s, _, loss_ = sess.run([all_summary, optimizer, loss], feed_dict={X: batch, Y: batch_ori})
			loss_all += loss_
		cost = loss_all / batch_size
		train_writer.add_summary(s, global_step=epoch + 1)
		print("Train Epoch: ", epoch + 1, ", Cost: ", cost)

		# test
		test_batch = int(len(test_img) / mini_batch) + 1
		all_psnr = 0
		for i in range(test_batch):
			if i == test_batch-1:
				batch = test_img[i*mini_batch:]
				batch_ori = test_ori[i*mini_batch:]
			else:
				batch = test_img[i*mini_batch:(i+1)*mini_batch]
				batch_ori = test_ori[i*mini_batch:(i+1)*mini_batch]
			s, psnr_ = sess.run([all_summary, psnr], feed_dict={X: batch, Y: batch_ori})
			all_psnr += psnr_
		psnr_avg = all_psnr / test_batch
		test_writer.add_summary(s, global_step=epoch + 1)
		print("Psnr: ", psnr_avg)

		if ( epoch + 1 ) % 100 == 0:
			print("Epoch: ", epoch + 1, ", Cost: ", cost)
			save(sess, ckpt_dir, epoch+1, saver)

	# validation
	print("validation")
	evalimg_list = np.array(os.listdir(eval_path))

	s = tf.InteractiveSession()
	for i in range(len(evalimg_list)):
		img_path = os.path.join(eval_path, evalimg_list[i])
		cur = Image.open(img_path)
		cur = tf.image.rgb_to_grayscale(cur).eval()

		h_size = (int(np.shape(cur)[0] / 2), int(np.shape(cur)[1] / 2))
		resized = tf.image.resize_images(cur, h_size, tf.image.ResizeMethod.BICUBIC)
		eval_img = tf.image.resize_images(resized, np.shape(cur)[0:2], tf.image.ResizeMethod.BICUBIC).eval()
		eval_img = np.reshape(eval_img, [1, eval_img.shape[0], eval_img.shape[1], eval_img.shape[2]])

		valid_result = model.build_model(eval_img)
		valid_result = tf.cast(valid_result, tf.uint8)
		with sess.as_default():
			valid_result = valid_result.eval()
			valid_result = np.reshape(valid_result, [cur.shape[0], cur.shape[1]])
		imgpath = os.path.join(result_path, str(i+1)+'.jpg')
		img = Image.fromarray(valid_result)
		img.save(imgpath)

# get result
input_path1 = "/Users/kimboyoung/Desktop/4-1/deep_learning/2019_ITE4053_2015004402/lab2/SR_dataset/91"
input_path2 = "/Users/kimboyoung/Desktop/4-1/deep_learning/2019_ITE4053_2015004402/lab2/SR_dataset/291"
config_path = "/Users/kimboyoung/Desktop/4-1/deep_learning/2019_ITE4053_2015004402/lab2"
get_result(input_path1, input_path2, config_path)