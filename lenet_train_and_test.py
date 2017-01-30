import time
import tensorflow as tf
import numpy as np
import pandas as pd
from scipy.misc import imread
from lenet import LeNet5
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# cifar 10
#from keras.datasets import cifar10
#(features, labels), (X_test, y_test) = cifar10.load_data()
#labels = labels[:,0]
#y_test = y_test[:,0]
#X_train,X_valid,y_train,y_valid = train_test_split(features,labels,test_size = 0.2,random_state = 0)

#nb_train = len(X_train)
#nb_classes = 10


# Load traffic sign
import pickle
training_file = "traffic-signs-data/train.p"
testing_file = "traffic-signs-data/test.p"

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

features, labels = train['features'], train['labels']
X_test, y_test = test['features'], test['labels']
X_train,X_valid,y_train,y_valid = train_test_split(features,labels,test_size = 0.2,random_state = 0)
nb_train = len(X_train)
nb_classes = 43



learning_rate = 0.001

X = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int64,(None))
keep_prob = tf.placeholder(tf.float32)

one_hot_y = tf.one_hot(y,depth=nb_classes)
logits = LeNet5(X,nb_classes)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits,one_hot_y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# evaluation
correct_prediction = tf.equal(tf.argmax(logits,1),tf.argmax(one_hot_y,1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]

        accuracy = sess.run(accuracy_operation, feed_dict={X: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples


saver = tf.train.Saver()

EPOCHS = 10
BATCH_SIZE = 128
# training
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    print("Training ...")
    for epoch in range(EPOCHS):

        X_train, y_train = shuffle(X_train, y_train)
        n_samples = len(X_train)

        # Loop over all batches
        for offset in range(0, n_samples, BATCH_SIZE):
            end = offset + BATCH_SIZE

            # Run optimization op (backprop) and cost op (to get loss value)
            sess.run(optimizer, feed_dict={X: X_train[offset:end], y: y_train[offset:end], keep_prob: 0.5})

        print("EPOCH {} ...".format(epoch + 1))
        validation_accuracy = evaluate(X_valid, y_valid)
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()

    print("Training finished.")

    saver.save(sess, 'lenet')
    print("Model saved")

with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))

    test_accuracy = evaluate(X_test, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))

