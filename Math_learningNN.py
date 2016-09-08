# Neural Net Algo (in Tensor Flow) for basic math learning
# --Author-- Rohan Hundia

import tensorflow as tf
import numpy as np
from DataSet import*
from Data_add import*
import matplotlib as mpl
import matplotlib.pyplot as plt


xinput = np.zeros((126,19))
yinput = np.zeros((126,18))
final_y = np.zeros((126,18))

np.set_printoptions(threshold=np.inf)


data = np.genfromtxt(r'DataSet_Add_Sub.csv',delimiter = '|' ,names = True, dtype = None,)

x_input = tf.placeholder(tf.float32,shape = [126,19], name = "X_Inputs")
y_input = tf.placeholder(tf.float32,shape = [126,18], name = "Y_Labels")


weight1 = tf.Variable(tf.zeros([19,18]), name = "Weight_Array")


bias1 = tf.Variable(tf.zeros([126,18]), name = "Bias1")

mat = tf.matmul(x_input,weight1)

final_y = (tf.nn.softmax(tf.matmul(x_input,weight1) + bias1)) #using logistic activation function


cost_function =  -tf.reduce_sum(y_input*tf.log(final_y)) #tf.reduce_mean(( (y_input * tf.log(final_y)) + ((1 - y_input) * tf.log(1.0 - final_y)) ) * -1)

alpha = 0.5 # determine learning rate of GD

train_step = tf.train.GradientDescentOptimizer(alpha).minimize(cost_function)

# Storing X and Y data from CSV file

for i in xrange (126):
    for j in xrange(1,38,2):
        xinput[i,round(j/2)] = float(data[i][0][j])
    for k in xrange(41,76,2):
        yinput[i,round((k-41)/2)] = float(data[i][0][k])


#xinput =  data_add()[0]
#yinput = data_add()[1]

init = tf.initialize_all_variables()
session = tf.Session()
session.run(init)

num_runs = 10000

for i in xrange(num_runs):
	session.run(train_step, feed_dict = {x_input: xinput, y_input: yinput})
	#cost = session.run(cost_function,feed_dict = {x_input: xinput, y_input:yinput})
	#plt.scatter([i],[cost], color = 'r')

	print("Run Number",i)
	print("Hypothesis", session.run(final_y,feed_dict={x_input: xinput, y_input: yinput}))
	#print("Weight1", session.run(weight1))
	#print("Bias1", session.run(bias1))



correct_array = tf.equal(tf.argmax(final_y,1), tf.argmax(y_input,1))
#print(session.run(correct_array,feed_dict={x_input: xinput, y_input:yinput}))
acc = tf.reduce_mean(tf.cast(correct_array, tf.float32))
print(session.run(acc, feed_dict={x_input: xinput, y_input:yinput}))
#print(session.run(cost_function,feed_dict = {x_input: xinput, y_input:yinput}))
#plt.ylabel('Cost')
#plt.xlabel('Epochs (Runs)')
#plt.show()
