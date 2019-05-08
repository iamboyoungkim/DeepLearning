import tensorflow as tf
import random
import math

sum = 0

def go():
	global sum
	x_data = []
	y_data = []

	total = 128
	for i in range(0, total*2):
		u = random.uniform(-1,1)
		v = random.uniform(-1,1)
		z = 0
		if u*u < v:
			z = 0
		else:
			z = 1
		x_data.append([u, v])
		y_data.append([z])

	# placeholders for a tensor that will be always fed.
	X = tf.placeholder(tf.float32, [None, 2])
	Y = tf.placeholder(tf.float32, [None, 1])

	W1 = tf.Variable(tf.random_normal([2, 2]), name='weight1')
	b1 = tf.Variable(tf.random_normal([2]), name='bias1')

	layer1 = tf.sigmoid(tf.matmul(X, W1) + b1)

	W2 = tf.Variable(tf.random_normal([2, 1]), name='weight2')
	b2 = tf.Variable(tf.random_normal([1]), name='bias2')

	hypothesis = tf.sigmoid(tf.matmul(layer1, W2) + b2)

	# cost/loss function
	cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))
	train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

	# Accuracy computation
	# True if hypothesis>0.5 else False
	predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
	accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

	# Launch graph
	with tf.Session() as sess:
		# Initialize TensorFlow variables
		sess.run(tf.global_variables_initializer())

		for step in range(5000):
			cost_val, _ = sess.run([cost, train], feed_dict={X: x_data[:128], Y: y_data[:128]})

# 			if step % 100 == 0:
# 				print(step, cost_val)

		# Accuracy report
		h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X: x_data[128:], Y: y_data[128:]})
		print("Accuracy: ", a)
		sum += a

for i in range(10):
	print(i+1, "번째 loop")
	go()

print("average accuracy : " , sum / 10)