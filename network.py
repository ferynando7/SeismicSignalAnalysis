#1.  (Input) -> [batch_size, 27, 1, 1] >> Apply 32 filter of [5x5]
#2.  (Convolutional layer 1) -> [batch_size, 27, 1, 32]
#3.  (ReLU 1) -> [?, 27, 1, 32]
#4.  (Max pooling 1) -> [?, 14, 1, 32]
#5.  (Convolutional layer 2) -> [?, 14, 1, 64]
#6.  (ReLU 2) -> [?, 14, 1, 64]
#7.  (Max pooling 2) -> [?, 7, 1, 64]
#8.  [fully connected layer 3] -> [1x150]
#9.  [ReLU 3] -> [1x150]
#10. [Drop out] -> [1x150] (optional)
#11. [fully connected layer 4] -> [1x10]



############### Start of the algorithm
import tensorflow as tf

# finish possible remaining session
sess.close()

#Start interactive session
sess = tf.InteractiveSession()



######### INPUT LAYER

##### Getting the data

data = #somehow get the data and prepare it

##### Parameters

width = 27 # width of the image in pixels 
height = 1 # height of the image in pixels
class_output = 9 # number of possible classifications for the problem

##### Placeholders
x  = tf.placeholder(tf.float32, shape=[None, width])
y_ = tf.placeholder(tf.float32, shape=[None, class_output])


#Convert data to tensors
x_tensor = tf.reshape([-1,27,1,1])


######### CONVOLUTIONAL LAYER 1

#Define kernels and biases

W_conv1 = tf.Variable(tf.truncated_normal([5, 1, 1, 32], stddev=0.1))
b_conv1 = tf.Variable(tf.constant(0.1, shape=[32])) # need 32 biases for 32 outputs

convolve1= tf.nn.conv2d(x_tensor, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1

######### RELU 1

h_conv1 = tf.nn.relu(convolve1)

######### MAX POOLING 1

conv1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1], padding='SAME') #max_pool_2x2

######### CONVOLUTIONAL LAYER 2

W_conv2 = tf.Variable(tf.truncated_normal([5, 1, 32, 64], stddev=0.1))
b_conv2 = tf.Variable(tf.constant(0.1, shape=[64])) #need 64 biases for 64 outputs

convolve2= tf.nn.conv2d(conv1, W_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2

######### RELU 2

h_conv2 = tf.nn.relu(convolve2)

######### MAX POOLING 2

conv2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1], padding='SAME') #max_pool_2x2

######### FULLY CONNECTED LAYER 3

# Flattening last layer

layer2_matrix = tf.reshape(conv2, [-1,7 * 64])

W_fc1 = tf.Variable(tf.truncated_normal([7 * 64, 150], stddev=0.1))
b_fc1 = tf.Variable(tf.constant(0.1, shape=[150])) # need 150 biases for 150 outputs

fcl = tf.matmul(layer2_matrix, W_fc1) + b_fc1

######### RELU 3

h_fc1 = tf.nn.relu(fcl)

######### SOFTMAX LAYER

W_fc2 = tf.Variable(tf.truncated_normal([150, class_output], stddev=0.1)) #1024 neurons
b_fc2 = tf.Variable(tf.constant(0.1, shape=[class_output])) # class_output = 9 possibilities for digits [1,2,3,4,5,6,7,8,9]

fc=tf.matmul(layer_drop, W_fc2) + b_fc2

y_CNN= tf.nn.softmax(fc)



########### DEFINE FUNCTIONS AND TRAIN ############

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_CNN), reduction_indices=[1]))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_CNN, 1), tf.argmax(y_, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


sess.run(tf.global_variables_initializer())


for i in range(1100):
    batch = ## put the data here
    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, float(train_accuracy)))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})