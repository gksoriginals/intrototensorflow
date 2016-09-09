"""
This is an example program classifying iris dataset using tensorflow API
Iris is one of the best dataset in pattern recognition
The dataset contain 3 classes of 50 types each,where each class refers to a type of iris plant
"""
import tensorflow as tf
#I am fetching the dataset from sklearn which is another machine learning API
from sklearn.datasets import load_iris
import numpy as np
data=load_iris()
"""
load_iris() return a dictionary that contains an array of names of flowers,a 150*4 array of features
and an array containg target label of each feature
"""
features=data['data']
print(features.shape)
#(150, 4)
print(features[:10])
"""
[[ 5.1  3.5  1.4  0.2]
 [ 4.9  3.   1.4  0.2]
 [ 4.7  3.2  1.3  0.2]
 [ 4.6  3.1  1.5  0.2]
 [ 5.   3.6  1.4  0.2]
 [ 5.4  3.9  1.7  0.4]
 [ 4.6  3.4  1.4  0.3]
 [ 5.   3.4  1.5  0.2]
 [ 4.4  2.9  1.4  0.2]
 [ 4.9  3.1  1.5  0.1]]
"""
labels=data['target']
print(labels[:10])
#[0 0 0 0 0 0 0 0 0 0]

print(labels[50:100])
#[1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
# 1 1 1 1 1 1 1 1 1 1 1 1 1]
#we need to shuffle the data

def randomize(dataset, labels):
  permutation = np.random.permutation(labels.shape[0])
  shuffled_dataset = dataset[permutation,:]
  shuffled_labels = labels[permutation]
  return shuffled_dataset, shuffled_labels
features,labels=randomize(features,labels)

print(labels[50:100])

"""
Now the features and labels are shuffled
[0 1 2 0 0 2 2 0 1 2 2 2 0 1 1 2 2 0 2 1 1 0 0 2 0 0 0 1 2 2 0 2 2 1 2 0 0
 2 2 0 1 1 2 1 0 1 2 1 0 1]

"""

#now i need a one-hot-encoder

def one_hot(dat):
    x=[0,0,0]
    labs=[]
    for i in dat:
        x[i]=1
        labs.append(x)
        x=[0,0,0]

    return labs

labels=one_hot(labels)

print(labels[50:100])
"""


#Here is the one-hot-encoded labels
[[0, 0, 1], [1, 0, 0], [1, 0, 0], [1, 0, 0], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 1, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 0, 1], [1, 0, 0], [0, 0, 1], [1, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1], [0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 0, 1], [0, 1, 0], [0, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1], [1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1], [0, 1, 0], [1, 0, 0]]
"""

"""
I am using the same algorithm used in MNIST classification here
"""

input=tf.placeholder(tf.float32,[None,4])
weights=tf.Variable(tf.zeros([4,3]))
bias=tf.Variable(tf.zeros([3]))
y=tf.nn.softmax(tf.matmul(input,weights)+bias)
y_=tf.placeholder(tf.float32,[None,3])
cross_entropy=-tf.reduce_sum(y_*tf.log(y))
train_step=tf.train.AdadeltaOptimizer(0.01).minimize(cross_entropy)
correct_prediction=tf.equal(tf.argmax(y,1),tf.arg_max(y_,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,"float"))
init=tf.initialize_all_variables()
sess=tf.Session()
sess.run(init)
train=features[:100]
lab_train=labels[:100]
lab_test=labels[100:]
test=features[100:]

for i in range(1000):
    sess.run(train_step,feed_dict={input:[x for x in train[:50]],y_:[x for x in lab_train[:50]]})
print(sess.run(accuracy,feed_dict={input:[x for x in test],y_:[x for x in lab_test]}))

#0.98



