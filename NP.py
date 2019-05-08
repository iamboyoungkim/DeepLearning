import random
import math
import numpy as np

sum = 0

def go():
	global sum
	dataset = []
	for i in range(0, 256):
		u = random.uniform(-1,1)
		v = random.uniform(-1,1)
		z = 0
		if u*u < v:
			z = 0
		else:
			z = 1
		dataset.append(((u, v), z))

	J = 0

	W1 = np.ones((2, 2))
	dW1 = np.zeros((2, 2))

	W2 = np.ones((1, 2))
	dW2 = np.zeros((1, 2))
	db1 = np.array([[0.0],[0.0]])
	db2 = np.array([0.0])

	b1 = np.array([[1], [1]])
	b2 = np.array([1])

	# sigmoid
	def sigmoid(gamma):
		if gamma < 0:
			return 1 - 1/(1 + math.exp(gamma))
		else:
			return 1/(1 + math.exp(-gamma))

	alp = 0.01
	testNum = 128.0
	for i in range(0, 5000):
		J = 0
		for now in range(0, 128):
			x = dataset[now][0][0]
			y = dataset[now][0][1]
			ans = dataset[now][1]

			X1 = np.array([[x], [y]])
			H1 = W1 @ X1 + b1

			x1 = sigmoid(H1[0][0])
			y1 = sigmoid(H1[1][0])

			X2 = np.array([[x1],[y1]])
			z1 = W2 @ X2 + b2
			z = z1[0]
			a = sigmoid(z)
			if a <= 0:
				J += 1
			elif a == 1:
				J += 0
			else:
				J += -(ans * math.log(a) + (1 - ans) * math.log(1 -a))/testNum

			dz = a - ans
			gW2 = dz
			dW2 += (gW2 / testNum) * X2.transpose()
			db2 += dz / testNum

			W2 = W2 - alp * dW2
			b2 = b2 - alp * db2

			dW1 += X1 @ (gW2 * W2) / testNum
			db1 += gW2 * W2.transpose()  / testNum

			W1 = W1 - alp * dW1
			b1 = b1 - alp * db1

		# print(J)

	correct = 0
	for now in range(128, 256):
		x = dataset[now][0][0]
		y = dataset[now][0][1]
		ans = dataset[now][1]

		X1 = np.array([[x], [y]])
		H1 = W1 @ X1 + b1

		x1 = sigmoid(H1[0][0])
		y1 = sigmoid(H1[1][0])

		X2 = np.array([[x1],[y1]])
		z1 = W2 @ X2 + b2
		z = z1[0]
		a = sigmoid(z)
		t = -1
		if z < 0.5:
			t = 0
		else:
			t = 1
		if t == ans:
			correct += 1

	print( "w1: ", W1, "w2: ", W2 )
	print("accuracy : " + str(float(correct)/float(128)))
	sum += float(correct)/float(128)

for i in range(10):
	print(i+1, "번째 loop")
	go()

print("average accuracy : " , sum / 10)