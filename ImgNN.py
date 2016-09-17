# Neural Net Algo (in Tensor Flow) for basic math learning
# --Author-- Rohan Hundiavised analytics to be done on this code)
# Autoencoder based NN for image learning at dot recognition.

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


data = np.genfromtxt(r'DataSet_Add_Sub.csv',delimiter = '|' ,names = True, dtype = None,)

x_input = tf.placeholder(tf.float32,shape = [None,27], name = "X_Inputs")
#y_input = tf.placeholder(tf.float32,shape = [None,18], name = "Y_Labels") 


weight1 = tf.Variable(tf.zeros([27,27]), name = "Weight1_Array") # for other data sets (Type1 and Type2) weights were initially set to 0
bias1 = tf.Variable(tf.zeros([2500,27]), name = "Bias1") 




#new_y = (tf.nn.softmax(tf.matmul(x_input,weight1) + bias1)) #using logistic activation function
final_y = (tf.nn.softmax(tf.matmul(x_input,weight1) + bias1))

cost_function =  -tf.reduce_sum(x_input*tf.log(final_y + 1e-10)) #-tf.reduce_sum(y_input*tf.log(final_y)) #tf.reduce_mean(( (y_input * tf.log(final_y)) + ((1 - y_input) * tf.log(1.0 - final_y)) ) * -1) 

alpha = 0.5 # p learning rate of GD 

train_step = tf.train.GradientDescentOptimizer(alpha).minimize(cost_function)

# Storing X and Y data from CSV file 
'''
for i in xrange (126):
    for j in xrange(1,38,2):
        xinput[i,round(j/2)] = float(data[i][0][j])
    for k in xrange(41,76,2):
        yinput[i,round((k-41)/2)] = float(data[i][0][k])
 '''
xinput = ImgCon().transpose()

init = tf.initialize_all_variables()
session = tf.Session()
session.run(init)

num_runs = 300

for i in xrange(num_runs):
	session.run(train_step, feed_dict = {x_input: xinput})
	cost = session.run(cost_function,feed_dict = {x_input: xinput})
	#print cost
	plt.scatter([i],[cost], color = 'r')
	
	#print("Run Number",i)
	#print("Hypothesis", session.run(final_y,feed_dict={x_input:xinput[1:64], y_input: yinput[1:64]}))
	#print("Weight2", session.run(weight2))
	#print("Bias2", session.run(bias2))
	if i == 200:
		gen_array = ("Hypothesis", session.run(final_y,feed_dict={x_input:xinput}))
	

	
#print(session.run(tf.argmax(final_y,1),feed_dict={x_input:xinput,y_input:yinput}))
correct_array = tf.equal(tf.argmax(final_y,1), tf.argmax(x_input,1))
acc = tf.reduce_mean(tf.cast(correct_array, tf.float32))
print(session.run(acc, feed_dict={x_input: xinput}))
#plt.ylabel('Cost')
#plt.xlabel('Epochs (Runs)')
#plt.show()

gen_array = np.array(gen_array[1], dtype = float)
gen_array = gen_array.transpose()
image2_array = (gen_array[25])*500
print("here")
image2_array = np.resize(image2_array,(50,50))
 
 
image2 = Image.fromarray(image2_array)
image2 = image2.convert('RGB')
image2.save("reproduced_image2.png")


