# Single layer autoencoder 
# Author : Rohan Hundia


import tensorflow as tf
import numpy as np 
from DataSet import*
from Data_add import*
from data3 import*
from imgload import*
import matplotlib as mpl
import matplotlib.pyplot as plt
import math


xinput = np.zeros((126,19))
yinput = np.zeros((126,18))
final_y = np.zeros((126,18))
new_y = []

np.set_printoptions(threshold=np.inf)


x_input = tf.placeholder(tf.float32, [None, 27])
weight1 = tf.Variable(tf.random_normal([27, 60],stddev=0.01))
bias1 = tf.Variable(tf.zeros([2500,60]))
weight2 = tf.Variable(tf.random_normal([60, 27],stddev=0.01))
bias2 = tf.Variable(tf.zeros([2500,27]))

hl1 = tf.nn.softmax(tf.matmul(x_input,weight1) + bias1)
final_y = tf.nn.softmax(tf.matmul(hl1,weight2) + bias2)


cost_function =  -tf.reduce_mean(x_input*tf.log(final_y + 1e-10)) #-tf.reduce_sum(y_input*tf.log(final_y)) #tf.reduce_mean(( (y_input * tf.log(final_y)) + ((1 - y_input) * tf.log(1.0 - final_y)) ) * -1) 

alpha = 0.5 #learning rate of GD 

train_step = tf.train.GradientDescentOptimizer(alpha).minimize(cost_function)

xinput = ImgCon().transpose()

init = tf.initialize_all_variables()
session = tf.Session()
session.run(init)

num_runs = 1000

for i in xrange(num_runs):
	print i
	session.run(train_step, feed_dict = {x_input: xinput})
	#print ("Train_Step:", session.run(train_step, feed_dict = {x_input: xinput}))
	cost = session.run(cost_function,feed_dict = {x_input: xinput})
	#print cost
	plt.scatter([i],[cost], color = 'r')
	if (i == (num_runs-1)):
		print("Hypothesis:",session.run(final_y,feed_dict={x_input:xinput}))
		weight_1 = (session.run(weight1))
		weight_2 =  (session.run(weight2))
		bias_1 = (session.run(bias1))
	correct_array = tf.equal(tf.argmax(final_y,1), tf.argmax(x_input,1))
	acc = tf.reduce_mean(tf.cast(correct_array, tf.float32))
	print(session.run(acc, feed_dict={x_input: xinput}))


correct_array = tf.equal(tf.argmax(final_y,1), tf.argmax(x_input,1))
print session.run(correct_array,feed_dict={x_input:xinput})
acc = tf.reduce_mean(tf.cast(correct_array, tf.float32))
print(session.run(acc, feed_dict={x_input: xinput}))

plt.show()

def softmax_manual(x):
	return np.exp(x) / np.sum(np.exp(x), axis=0)

val_3_input = ImgCon()[0:3].transpose()
weight1_val_3 = (np.array(weight_1))[0:3]
weight2_val_3 = (np.array(weight_2))
bias_1 = np.array(bias_1)


#print(softmax_manual(np.dot(val_3_input,weight1_val_3)+ bias_1))
layer_1_val_3 = (np.dot(val_3_input,weight1_val_3)+bias_1).transpose()
#layer2_val_3 = (np.dot(layer_1_val_3,weight2_val_3)).transpose()

print len(layer_1_val_3)
sum_array,x = [],[]
for k in xrange(len(layer_1_val_3)):
	sum_array.append(np.sum(layer_1_val_3[k]))
for i in xrange(61):
	if i != 0:
		x.append(i)
print len(sum_array)
plt.bar(x,sum_array)
plt.show()
