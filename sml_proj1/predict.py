import pandas as pd
import numpy as np
import math

features = pd.read_csv("features_data_30.csv")


training_Y = features['label'][0:-2000]
l = len(training_Y)
training_Y = np.array(training_Y).reshape(l,)


def load_in_training_features(lst):
    
    first = 1
    for f in lst:
        tmp_f = f[0:-2000]
        l = len(tmp_f)
        tmp_f = np.array(tmp_f).reshape(l,1)
        if first:
            training_X = tmp_f
            first = 0
        else:
            training_X = np.append(training_X, tmp_f, axis = 1)
        
    return training_X

def load_in_predict_features(lst):
    
    first = 1
    for f in lst:
        tmp_f = f[-2000::]
        l = len(tmp_f)
        tmp_f = np.array(tmp_f).reshape(l,1)
        if first:
            testing_X = tmp_f
            first = 0
        else:
            testing_X = np.append(testing_X, tmp_f, axis = 1)
        
    return testing_X

def normalise(feature):
    result = []
    for num in feature:
        result.append(math.log(num + 1))
    return result


################### features ####################
f1 = features['f1']
f2 = features['f2']
f3 = normalise(features['f3'])
f4 = features['f4']
f5 = features['f5']

feature_list = [f1,f2,f3,f4,f5]

training_X = load_in_training_features(feature_list)
testing_X = load_in_predict_features(feature_list)



class DatasetIterator:

    def __init__(self, features, labels, batch_size):
        assert(features.shape[0]==labels.shape[0])
        assert(batch_size > 0 and batch_size <= features.shape[0])
        self.features = features
        self.labels = labels
        self.num_instances = features.shape[0]
        self.batch_size = batch_size
        self.num_batches = self.num_instances//self.batch_size  #only get the integer part
        if (self.num_instances%self.batch_size!=0):
            self.num_batches += 1                               #for the reminder part
        self._i = 0
        self._rand_ids = None

    def __iter__(self):
        self._i = 0
        self._rand_ids = np.random.permutation(self.num_instances)
        return self
        
    def __next__(self):
        if self.num_instances - self._i >= self.batch_size:
            this_rand_ids = self._rand_ids[self._i:self._i + self.batch_size]
            self._i += self.batch_size
            return self.features[this_rand_ids], self.labels[this_rand_ids]
        elif self.num_instances - self._i > 0:
            this_rand_ids = self._rand_ids[self._i::]
            self._i = self.num_instances
            return self.features[this_rand_ids], self.labels[this_rand_ids]
        else:
            raise StopIteration()
            
batch_size = 100
X = training_X

# change Y to one_hot
Y = np.zeros([len(training_Y), 2])
for i in range(len(training_Y)):
    if training_Y[i] == 1:
        Y[i][1] = 1
    else:
        Y[i][0] = 1


test_set = 2000
        
# randomly choice a test set
test_id = np.random.choice(len(training_Y),test_set,replace=False)
train_id = list(set(range(len(training_Y))).difference(set(test_id)))
x_train = X[train_id]
y_train = Y[train_id]
x_test = X[test_id]
y_test = Y[test_id]
y_train_label = training_Y[train_id]
y_test_label = training_Y[test_id]

train_iterator = DatasetIterator(x_train, y_train, batch_size)



###################### nerual network model #########################


import tensorflow as tf

n_input = len(feature_list)
n_hidden = 200
n_classes = 2

hold = None

def mlp_config(n_input, n_hidden, n_classes):
    x = tf.placeholder(tf.float32, [None, n_input], name = 'x')
    y = tf.placeholder(tf.uint8, [None, n_classes], name = 'y')
    
    weights = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden])),
        'h2': tf.Variable(tf.random_normal([n_hidden, n_hidden])),
        'h3': tf.Variable(tf.random_normal([n_hidden, n_hidden])),
        'out': tf.Variable(tf.random_normal([n_hidden, n_classes])),
    }
    
    bias = {
        'h1':tf.Variable(tf.random_normal([n_hidden])),
        'h2':tf.Variable(tf.random_normal([n_hidden])),
        'h3':tf.Variable(tf.random_normal([n_hidden])),
        'out':tf.Variable(tf.random_normal([n_classes]))
    }
    
    return x, y, weights, bias


def mlp_model(x, y, weights, bias):
    
    
    hidden1 = tf.nn.sigmoid(tf.matmul(x, weights['h1']) + bias['h1'])
    hidden2 = tf.nn.sigmoid(tf.matmul(hidden1, weights['h2']) + bias['h2'])
    hidden3 = tf.nn.sigmoid(tf.matmul(hidden2, weights['h3']) + bias['h3'])   
    
    logits = tf.matmul(hidden3, weights['out']) + bias['out']
    pred = tf.one_hot(tf.cast(tf.argmax(logits,1), tf.int32), depth = 2)
    prob = tf.nn.softmax(logits)
    return pred, logits, prob

def get_loss(logits, y):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = y, logits = logits))
    return loss

def get_accuracy(pred, y):
    corr_pre = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(corr_pre, tf.float32))
    return accuracy


def train_model():
    x,y,weights,bias = mlp_config(n_input, n_hidden, n_classes)
    pred, logits, prob= mlp_model(x, y, weights, bias)
    loss = get_loss(logits, y)
    accuracy = get_accuracy(pred, y)
    
    train_step = tf.train.GradientDescentOptimizer(learning_rate = 0.01).minimize(loss)
    sess = tf.Session()
    init = tf.global_variables_initializer()
    
    saver = tf.train.Saver()
    sess.run(init)
    
    for epoch in range(50):
        for X_batch, Y_batch in train_iterator:
            _,l = sess.run([train_step, loss], feed_dict={x: X_batch, y: Y_batch})
        print(str(epoch) +' loss: ' + str(l))
    
    print('accuracy: ')
    acc, predprob = sess.run([accuracy,prob], feed_dict = {x: x_test, y: y_test})
    print(acc)
    
    prob = sess.run(prob, feed_dict = {x: testing_X})
    
    f = open('output.csv','w')
    f.write('Id,Prediction\n')
    for i in range(len(prob)):
        f.write(str(i+1) + ',' + str(prob[i][1]) + '\n')
    f.close()
    
    
train_model()