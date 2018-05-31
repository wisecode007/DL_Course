
# coding: utf-8

# In[5]:


import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
import numpy as np
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
n_input = 784
n_output = 10
net_input = tf.placeholder(tf.float32, [None, n_input])
y_true = tf.placeholder(tf.float32, [None, 10])

hidden_size = 256
W1 = tf.Variable(tf.truncated_normal([n_input, hidden_size]))
b1 = tf.Variable(tf.truncated_normal([hidden_size]))
W2 = tf.Variable(tf.truncated_normal([hidden_size, n_output]))
b2 = tf.Variable(tf.truncated_normal([n_output]))
net_output = tf.nn.relu(tf.matmul(net_input, W1) + b1)  
net_output = tf.nn.dropout(net_output, 0.525) 

#final layer
net_output = (tf.matmul(net_output, W2) + b2) 





correct_prediction = tf.equal(tf.argmax(net_output, 1), tf.argmax(y_true, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
cost = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=net_output, labels=y_true))

eta = 0.0125 #values [0.0001,...,0.1]

optimizer = tf.train.AdamOptimizer(eta).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
l_loss = list()
l_loss2 = list()

batch_size = 350  
n_epochs = 12  
for epoch_i in range(n_epochs):
    for batch_i in range(0, mnist.train.num_examples, batch_size):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        sess.run(optimizer, feed_dict={
            net_input: batch_xs,
            y_true: batch_ys
        })
        
    loss = sess.run(accuracy, feed_dict={
                       net_input: mnist.validation.images,
                       y_true: mnist.validation.labels })
    loss2 = sess.run(accuracy, feed_dict={
                       net_input: mnist.train.images,
                       y_true: mnist.train.labels })
    print('Validation accuracy for epoch {} is: {}'.format(epoch_i + 1, loss))
    print('Train accuracy for epoch {} is: {}'.format(epoch_i + 1, loss2))
    l_loss.append(loss)
    l_loss2.append(loss2)



plt.title('NN Acuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.plot(l_loss, color='m')  #validation loss
plt.plot(l_loss2, color='g') #train loss
plt.show()

print ('Test resualt:')
print(sess.run(accuracy, feed_dict={net_input: mnist.test.images, y_true: mnist.test.labels}))
    

