import matplotlib.pyplot as plt
import numpy as np
import random
import tensorflow as tf
import sys
from utils import *
from sklearn.utils import shuffle

#################################################################################
#			MAD - Exercise set 2					#
#			Neural Networks         				#
#           		Regression with NN                      		#
#################################################################################

# In this exercise, we offer you a template to perform regression on a simple sinusoidal signal. You have to fill in the missing parts, which are the definition of the model, the loss and the trainer. In forthcoming exercises, the template will be much less revealing. The objective of this exercise is to understand the functionality of tensorflow and the training procedure.

# The data in this exercise is divided in two sets, the training and validation. Please consult the note at the end of the template to understand the reasoning behind this.




# Creating the sinusoidal data for training
N = 200
noise = 0.0
data_input = np.linspace(0,2*np.pi, N) + noise * np.random.randn(N)
data_targets = np.sin(data_input) + noise * np.random.randn(data_input.shape[0])
plot(data_input, data_targets, 'train_data.pdf')

# divide data in training and validation
data_input, data_targets = shuffle(data_input, data_targets, random_state=0)

# First half of the data for training, second half for validation
train_input = data_input[:int(N/2)]
train_targets = data_targets[:int(N/2)]
val_input = data_input[int(N/2):]
val_targets = data_targets[int(N/2):]

# Testing data, the real sinusoidal signal
test_input = np.linspace(0,2*np.pi, 1000)
test_targets = np.sin(test_input)
plot(test_input, test_targets, 'sinus.pdf')

# Fixing the dimensions
train_input = np.reshape(train_input, (-1, 1))
train_targets = np.reshape(train_targets, (-1, 1))
val_input = np.reshape(val_input, (-1, 1))
val_targets = np.reshape(val_targets, (-1, 1))
test_input = np.reshape(test_input, (-1, 1))
test_targets = np.reshape(test_targets, (-1, 1))




# Building the model
'''
TODO: Build a neural network with one intermediate layer with $h=10$ hidden units and $\textbf{tanh}$ activation function, to solve the task. Do not use an activation in the output (identity). Bare in mind that the batch size is an additional dimension in the tensorflow model. Use the variable name output for the final model output.
You need to define placeholders for the input and the targets. Then define a neural network model, with one intermediate layer that connects the input to the output. The final output of the neural network is provided.
'''


output = tf.einsum('kh,hc->kc', layer, W2) + b2 # or equivalently output = tf.matmul(layer, W2) + b2 , many choices possible


'''
TODO: Define the MSE loss function (variable: rmse_loss, is a function of a placeholder with the name targets and the output) and select a tensorflow trainer of your choice (trainer_rmse). e.g. tf.train.GradientDescentOptimizer(learning_rate).minimize(rmse_loss).
Then set the noise level to zero, set the batch size to 10 and run 100000 epochs. Can you recover approximately the sinusoidal signal? Plot the result and compare it qualitatively. Try different learning rates and different optimizers. Search online for available optimizers in tensorflow.

'''




# Training loop:

n_samples = train_input.shape[0]
batch_size = 10
number_of_batches = n_samples//batch_size
num_epochs = 10000

train_loss = []
val_loss = []
print('Number of batches: {:d}'.format(number_of_batches))
with tf.Session() as sess:
    # Initialize all variables
    tf.global_variables_initializer().run()
    test_rmse = sess.run(rmse_loss, feed_dict={input:test_input, target:test_targets})
    print('Before training, Test RMSE {:.5f}'.format(test_rmse))
    for epoch in range(num_epochs):
        for i in range(number_of_batches):
            print('Batch {:d}/{:d}'.format(i, number_of_batches))
            sys.stdout.write("\033[F")

            batch_input = getBatch(train_input, i, batch_size)
            batch_target = getBatch(train_targets, i, batch_size)

            _, batch_loss = sess.run([trainer_rmse, rmse_loss], feed_dict={input:batch_input, target: batch_target})

        train_rmse = sess.run(rmse_loss, feed_dict={input:train_input, target:train_targets})
        val_rmse = sess.run(rmse_loss, feed_dict={input:val_input, target:val_targets})

        train_loss.append(train_rmse)
        val_loss.append(val_rmse)

        print('Epoch {:d}/{:d}, RMSE on Train data set: {:.5f}%, RMSE on Validation data set: {:.5f}%'.format(epoch, num_epochs, train_rmse, val_rmse))

    output_test, rmse_loss = sess.run([output, rmse_loss], feed_dict={input:test_input, target:test_targets})

    print('RMSE on TEST data set: {:.5f}%'.format(rmse_loss))


    plotSinusTest(test_input, output_test, test_targets, 'nn_sin.pdf')
    plotLosses(train_loss, val_loss, 'losses.pdf')






# Note: In all examples considered in this Homework we did not cover overfitting as the data set is simple. In the real world, data might be noisy, scarce or the model we select might be too expressive. Overfitting is when a model captures characteristics of the training data set that do not generalize, i.e. they do not help us infer on a different data set. Imagine for example that we have data about nutrition from a specific group of people $P_1$ depending on their age, habits, and professions and we want to predict what nutrition habits a different group $P_2$ has. If we train on $P_1$, we may capture characteristics that are not generalizable to all people but rather to this specific group. One way to tackle this problem is to shuffle $P_1$ and randomly divide it in two groups $P_1^{train}$ and  $P_1^{val}$. Then, we train the NN only on $P_1^{train}$ and track the loss on $P_1^{val}$. Since during training we minimize the loss on $P_1^{train}$, as we train for multiple epochs the loss will be decreasing (not necessarily monotonically). However, if the two data sets have differences and our model is expressive enough, after some time, the loss in the validation data set will start increasing. We stop the training procedure after detecting this epoch. This method of avoiding overfitting is called early stopping.
