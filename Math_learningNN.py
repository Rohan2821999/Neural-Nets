# Neural Net Algo (in Tensor Flow) for basic math learning
# --Author-- Rohan Hundia

import tensorflow as tf
import numpy as np 

xinput = np.zeros((126,19))
yinput = np.zeros((126,18))

data = np.genfromtxt(r'DataSet_Add_Sub.csv',delimiter = '|' ,names = True, dtype = None,)

x_input = tf.placeholder(tf.float32,shape = [126,19], name = "X_Inputs")
y_input = tf.placeholder(tf.float32,shape = [126,18], name = "Y_Labels") 

weight1 = tf.Variable(tf.random_uniform([19,18],0,1), name = "Weight_Array") 

bias1 = tf.Variable(tf.zeros([18]), name = "Bias1") 

final_y = tf.nn.sigmoid(tf.matmul(x_input,weight1) + bias1) #using logistic activation function

cost_function = tf.reduce_mean(-tf.reduce_sum(final_y * tf.log(y_input), reduction_indices=[1]))

alpha = 0.05 # determine learning rate of GD 

train_step = tf.train.GradientDescentOptimizer(alpha).minimize(cost_function)

# Storing X and Y data from CSV file 
for i in xrange (126):
    for j in xrange(1,38,2):
        xinput[i,round(j/2)] = float(data[i][0][j])
    for k in xrange(41,76,2):
        yinput[i,round((k-41)/2)] = float(data[i][0][k])

init = tf.initialize_all_variables()
session = tf.Session()
session.run(init)

num_runs = 100

for i in xrange(num_runs):
	session.run(train_step, feed_dict = {x_input: xinput, y_input: yinput})
	print("Run Number",i)
	print("Hypothesis", session.run(final_y,feed_dict={x_input: xinput, y_input: yinput}))
	print("Weight1", session.run(weight1))
	print("Bias1", session.run(bias1))
	

	
#correct_array = tf.equal(tf.argmax(final_y,1), tf.argmax(y_input,1))
#acc = tf.reduce_mean(tf.cast(correct_array, tf.float32))
#print(session.run(acc, feed_dict={x_input: xinput, y_input:yinput}))
