# coding: utf-8
# 2018-08-17

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

# save and load
import pickle # import cpickle as pickle

def data_save(data,filename):
    f = open(filename, "wb") # *.dat
    pickle.dump(data, f)
    f.close()

def data_load(filename):
    return pickle.load(open(filename, "rb"))

# # Model

class Model(object):

    def __init__(self, **args):
        """ Build some model here """
        print(args)

    def predict(inputs):
        raise NotImplementedError
        
    def loss():
        raise NotImplementedError
    
    def metrics():
        raise NotImplementedError
        
    def optimizer():
        raise NotImplementedError

    def train(dataset):
        raise NotImplementedError
    
    def set_tensorboard(self,logdir):
        self.tensorboard_dir = logdir

# ## Model_mnist

class Model_mnist(Model):    
    name = 'mnist'
    method = 'none'
    #image_size = 28
    image_channel = 1
        
    #def __init__(self, **args):
    def __init__(self, image_size=28, hidden_size=100, **args):

        tf.reset_default_graph()
        self.image_size = image_size
        self.hidden_size = hidden_size
        
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')
        learning_rate_abph = tf.placeholder(tf.float32, name='learning_rate_ab')
        is_training = tf.placeholder(tf.bool, name='is_training')
        
        #x = tf.placeholder(tf.float32, shape=[None, 784], name='x')
        x = tf.placeholder(tf.float32, shape=[None, self.image_channel * self.image_size**2], name='x')
        labels = tf.placeholder(tf.float32, shape=[None, 10], name='labels')
        #x_image = tf.reshape(x, [-1, 28, 28, 1])
        x_image = tf.reshape(x, [-1, self.image_size, self.image_size, self.image_channel])
        
        self.x = x
        self.labels = labels
        self.learning_rate = learning_rate
        self.learning_rate_abph = learning_rate_abph
        self.learning_rate_ab = 0.1 # default
        self.is_training = is_training
        
        self.predict(x_image)
        self.loss(labels)
        self.metrics()
        self.optimizer()
        
        self.init = tf.global_variables_initializer()
        self.saver = tf.train.Saver(max_to_keep=0)
        
        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('accu', self.accuracy)
        
        print('A model for %s is created using %s method.' %(self.name, self.method))
        
        self.scaling = [] # add to test the scaling property
        self.regamma = [] # add to test different value of gamma
    
    def __del__(self):
        print("__del__")
        
    def predict(self, x_image):
        raise NotImplementedError
        
    def loss(self, labels):
        y = self.y
        self.labels = labels
        #cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits))
        cross_entropy = -tf.reduce_mean(tf.reduce_sum(labels * tf.log(y), reduction_indices=[1]),name="cross_entropy")
        self.loss = cross_entropy
    
    def metrics(self):
        y = self.y
        labels = self.labels
        correct_prediction = tf.equal(tf.argmax(y, axis=1), tf.argmax(labels, axis=1), name='correct_prediction')
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32),name='accuracy')        
        self.accuracy = accuracy
    
    def optimizer(self):
        learning_rate = self.learning_rate
        #self.training_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.loss)
        
        with tf.name_scope("train"):
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            print('extra_update_ops:\n',extra_update_ops)
            with tf.control_dependencies(extra_update_ops):
                self.training_op = optimizer.minimize(self.loss) 
    
    def set_scaling(self,weight_aug=1.0):
        # a test 
        
        self.scaling = []
        #print(tf.trainable_variables())
        print('\n')
        for w in tf.trainable_variables():
            if 'gamma' in w.name or 'beta' in w.name:
                print('not scaling:',w)
            else:
                print('scaling %g:' % weight_aug,w)
                self.scaling.append( tf.assign(w, w*weight_aug) )
                
    def set_gamma(self,gamma=1.0):
        
        self.regamma = []
        for w in tf.trainable_variables():
            if 'gamma' in w.name:
                print('set value %g:' % gamma,w)
                self.regamma.append( tf.assign(w, gamma) )
        
    def train(self, dataset, learning_rate = 1e-3, n_batch=100, max_step=1000):
        
        #sess = tf.InteractiveSession()        
        with tf.Session() as sess:
            
            sess.run(self.init)
            sess.run(self.scaling) # test the scaling property
            sess.run(self.regamma) # test the gamma value

            plan_tag =  '%s_lr%g_nb%d_it%d' % (self.method, learning_rate, n_batch, max_step)
            print(plan_tag)
            
            tensorboard_dir = self.tensorboard_dir
            saver = self.saver  #.as_saver_def()
            saver_path = tensorboard_dir + plan_tag + '_par/'
            if not os.path.exists(tensorboard_dir):
                os.makedirs(tensorboard_dir)
                
            '''writer = tf.summary.FileWriter(tensorboard_dir + plan_tag + '_train')
            writer_test = tf.summary.FileWriter(tensorboard_dir + plan_tag + '_test')
            merged_summary = tf.summary.merge_all()
            writer.add_graph(sess.graph)'''
            # RuntimeError: Graph is finalized and cannot be modified.
            
            sess.graph.finalize() # RuntimeError: Graph is finalized and cannot be modified.
            
            value_history = []
            for i in range(max_step+1):

                xb,yb = dataset.train.next_batch(n_batch)
                feed_dict_train = {self.x:xb, self.labels:yb, self.is_training:True,
                                   self.learning_rate:learning_rate,
                                   self.learning_rate_abph:self.learning_rate_ab}
                sess.run(self.training_op, feed_dict=feed_dict_train)

                if i%10 == 0:

                    try:
                        xt,yt = dataset.test.next_batch(n_batch)
                        feed_dict_test = {self.x:xt, self.labels:yt, self.is_training:False}

                        #train_loss, train_accu = sess.run((self.loss, self.accuracy),feed_dict=feed_dict_train)
                        #test_loss, test_accu = sess.run((self.loss,self.accuracy), feed_dict=feed_dict_test)
                        
                        train_loss = sess.run(self.loss,feed_dict=feed_dict_train)
                        train_accu = sess.run(self.accuracy,feed_dict=feed_dict_train)
                        test_loss = sess.run(self.loss, feed_dict=feed_dict_test)
                        test_accu = sess.run(self.accuracy, feed_dict=feed_dict_test)

                        #value_history.append([train_loss, train_accu, test_loss, test_accu])
                        value_history.append([i,train_loss, train_accu, test_loss, test_accu])

                        print('%d : train_loss = %g, test_err = %g, train_accu = %g, test_accu = %g'
                              % (i,train_loss, test_loss, train_accu,test_accu))
                        
                        '''s = sess.run(merged_summary, feed_dict=feed_dict_train)
                        writer.add_summary(s,i)

                        st = sess.run(merged_summary, feed_dict=feed_dict_test)
                        writer_test.add_summary(st,i)'''

                        #saver.save(sess, saver_path, global_step=i )
                        
                        if train_loss != train_loss:
                            break
                            
                    except:
                        break
                        
            saver.save(sess, saver_path, global_step=i )
            
            self.datafile = tensorboard_dir + plan_tag+'.dat'
            data_save(np.array(value_history),filename=self.datafile) # save

            self.value_history = value_history
            sess.close()


# # Model1 -- 2cnn + 2fc 

class Model_mnist_gd(Model_mnist):
    method = 'gd'
    
    def predict(self, x_image):
        
        with tf.variable_scope(self.method):

            layer1 = tf.layers.conv2d(x_image, 32, kernel_size=[5,5],strides=[1,1],padding='SAME',
                                      activation=tf.nn.relu, name='layer1')
            pool1 = tf.nn.max_pool(layer1, ksize=[1,2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

            layer2 = tf.layers.conv2d(pool1, 64, kernel_size=[5,5],strides=[1,1],padding='SAME',
                                      activation=tf.nn.relu, name='layer2')
            pool2 = tf.nn.max_pool(layer2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

            flat_shape = pool2.get_shape()[1:4].num_elements() # 7*7*64 = 3136
            flattened = tf.reshape(pool2, [-1, flat_shape])

            fc1 = tf.layers.dense(flattened,1024, activation=tf.nn.relu, name='fc1')
            logits = tf.layers.dense(fc1,10,activation=None, name='fc2')

            tf.summary.histogram('logits', logits)

            
        self.logits = logits
        self.y = tf.nn.softmax(logits)

class Model_mnist_bn(Model_mnist):
    method = 'bn'
    
    def predict(self, x_image):
        
        with tf.variable_scope(self.method):

            hidden1 = tf.layers.conv2d(x_image, 32, kernel_size=[5,5],strides=[1,1],padding='SAME',
                                       activation=None, name='hidden1')
            bn1 = tf.layers.batch_normalization(hidden1,training=self.is_training, momentum=0.9, name='bn1')
            layer1 = tf.nn.relu(bn1, name='layer1')
            pool1 = tf.nn.max_pool(bn1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')

            hidden2 = tf.layers.conv2d(pool1, 64, kernel_size=[5,5],strides=[1,1],padding='SAME',
                                       activation=None, name='hidden2')
            bn2 = tf.layers.batch_normalization(hidden2,training=self.is_training, momentum=0.9, name='bn2')
            layer2 = tf.nn.relu(bn1)
            pool2 = tf.nn.max_pool(layer2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

            flat_shape = pool2.get_shape()[1:4].num_elements() # 7*7*64 = 3136
            flattened = tf.reshape(pool2, [-1, flat_shape])

            hidden3 = tf.layers.dense(flattened,1024,activation=None, name='fc1')
            bn3 = tf.layers.batch_normalization(hidden3,training=self.is_training, momentum=0.9, name='bn3')
            fc1 = tf.nn.relu(bn3)

            hidden4 = tf.layers.dense(fc1,10,activation=None, name='fc2')
            logits = tf.layers.batch_normalization(hidden4,training=self.is_training, momentum=0.9, name='bn4')  
            
            tf.summary.histogram('logits', logits)
        
        self.logits = logits
        self.y = tf.nn.softmax(logits)

class Model_mnist_bn_split(Model_mnist_bn):
    method = 'bn_split'
    
    def optimizer(self):
        
        #self.training_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.loss)
        
        learning_rate_ab = self.learning_rate_abph
        learning_rate = self.learning_rate
        
        list0 = tf.trainable_variables()
        list2 = tf.trainable_variables(scope='bn_split/bn')
        list1 = list(set(list0)-set(list2))
        print(list1,list2)
        
        with tf.name_scope("train"):
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            #self.training_op = optimizer.minimize(self.loss)
            self.training_op1 = optimizer.minimize(self.loss, var_list=list1)
            
            optimizer2 = tf.train.GradientDescentOptimizer(learning_rate=learning_rate_ab)
            extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(extra_update_ops):
                self.training_op2 = optimizer2.minimize(self.loss, var_list=list2)
            self.training_op = (self.training_op1,self.training_op2)

# # Model2 -- 1fc -- one layer

class Model2_mnist_gd(Model_mnist):
    method = 'gd'
    
    def predict(self, x_image):
        
        x = self.x # not use x_image
        with tf.variable_scope(self.method):

            #flattened = tf.reshape(x, [-1, 28*28])
            flattened = tf.reshape(x, [-1, self.image_size*self.image_size])
            logits = tf.layers.dense(flattened,10,activation=None, name='fc')

            tf.summary.histogram('logits', logits)

            
        self.logits = logits
        self.y = tf.nn.softmax(logits)

class Model2_mnist_bn(Model_mnist):
    method = 'bn'
    
    def predict(self, x_image):
        
        x = self.x # not use x_image
        with tf.variable_scope(self.method):

            #flattened = tf.reshape(x, [-1, 28*28])
            flattened = tf.reshape(x, [-1, self.image_size*self.image_size])
            hidden = tf.layers.dense(flattened,10,activation=None, name='fc')
            
            logits = tf.layers.batch_normalization(hidden,training=self.is_training, momentum=0.9, name='bn')  
            
            tf.summary.histogram('logits', logits)
        
        self.logits = logits
        self.y = tf.nn.softmax(logits)

class Model2_mnist_bn_split(Model2_mnist_bn):
    method = 'bn_split'
    
    def optimizer(self):
        
        #self.training_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.loss)
        
        learning_rate_ab = self.learning_rate_abph
        learning_rate = self.learning_rate
        
        list0 = tf.trainable_variables()
        list2 = tf.trainable_variables(scope='bn_split/bn')
        list1 = list(set(list0)-set(list2))
        print(list1,list2)
        
        with tf.name_scope("train"):
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            #self.training_op = optimizer.minimize(self.loss)
            self.training_op1 = optimizer.minimize(self.loss, var_list=list1)
            
            optimizer2 = tf.train.GradientDescentOptimizer(learning_rate=learning_rate_ab)
            extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(extra_update_ops):
                self.training_op2 = optimizer2.minimize(self.loss, var_list=list2)
            self.training_op = (self.training_op1,self.training_op2)

# # Model3 -- 2cnn(3) + 3fc 

class Model3_mnist_gd(Model_mnist):
    method = 'gd'
    
    def predict(self, x_image):
        
        with tf.variable_scope(self.method):

            layer1 = tf.layers.conv2d(x_image, 32, kernel_size=[3,3],strides=[1,1],padding='SAME',
                                      activation=tf.nn.relu, name='layer1')
            pool1 = tf.nn.max_pool(layer1, ksize=[1,2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

            layer2 = tf.layers.conv2d(pool1, 64, kernel_size=[3,3],strides=[1,1],padding='SAME',
                                      activation=tf.nn.relu, name='layer2')
            pool2 = tf.nn.max_pool(layer2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

            flat_shape = pool2.get_shape()[1:4].num_elements() # 7*7*64 = 3136
            flattened = tf.reshape(pool2, [-1, flat_shape])

            fc1 = tf.layers.dense(flattened,512, activation=tf.nn.relu, name='fc1')
            fc2 = tf.layers.dense(fc1,128, activation=tf.nn.relu, name='fc2')
            logits = tf.layers.dense(fc2,10,activation=None, name='fc3')

            tf.summary.histogram('logits', logits)

            
        self.logits = logits
        self.y = tf.nn.softmax(logits)

class Model3_mnist_bn(Model_mnist):
    method = 'bn'
    
    def predict(self, x_image):
        
        with tf.variable_scope(self.method):

            hidden1 = tf.layers.conv2d(x_image, 32, kernel_size=[3,3],strides=[1,1],padding='SAME',
                                       activation=None, name='hidden1')
            bn1 = tf.layers.batch_normalization(hidden1,training=self.is_training, momentum=0.9, name='bn1')
            layer1 = tf.nn.relu(bn1, name='layer1')
            pool1 = tf.nn.max_pool(bn1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')

            hidden2 = tf.layers.conv2d(pool1, 64, kernel_size=[3,3],strides=[1,1],padding='SAME',
                                       activation=None, name='hidden2')
            bn2 = tf.layers.batch_normalization(hidden2,training=self.is_training, momentum=0.9, name='bn2')
            layer2 = tf.nn.relu(bn1)
            pool2 = tf.nn.max_pool(layer2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

            flat_shape = pool2.get_shape()[1:4].num_elements() # 7*7*64 = 3136
            flattened = tf.reshape(pool2, [-1, flat_shape])

            hidden3 = tf.layers.dense(flattened,512,activation=None, name='fc1')
            bn3 = tf.layers.batch_normalization(hidden3,training=self.is_training, momentum=0.9, name='bn3')
            fc1 = tf.nn.relu(bn3)
            
            hidden4 = tf.layers.dense(fc1,128,activation=None, name='fc2')
            bn4 = tf.layers.batch_normalization(hidden4,training=self.is_training, momentum=0.9, name='bn4')
            fc2 = tf.nn.relu(bn4)

            hidden5 = tf.layers.dense(fc2,10,activation=None, name='fc3')
            logits = tf.layers.batch_normalization(hidden5,training=self.is_training, momentum=0.9, name='bn5')  
            
            tf.summary.histogram('logits', logits)
        
        self.logits = logits
        self.y = tf.nn.softmax(logits)

class Model3_mnist_bn_split(Model3_mnist_bn):
    method = 'bn_split'
    
    def optimizer(self):
        
        #self.training_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.loss)
        
        learning_rate_ab = self.learning_rate_abph
        learning_rate = self.learning_rate
        
        list0 = tf.trainable_variables()
        list2 = tf.trainable_variables(scope='bn_split/bn')
        list1 = list(set(list0)-set(list2))
        print(list1,list2)
        
        with tf.name_scope("train"):
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            #self.training_op = optimizer.minimize(self.loss)
            self.training_op1 = optimizer.minimize(self.loss, var_list=list1)
            
            optimizer2 = tf.train.GradientDescentOptimizer(learning_rate=learning_rate_ab)
            extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(extra_update_ops):
                self.training_op2 = optimizer2.minimize(self.loss, var_list=list2)
            self.training_op = (self.training_op1,self.training_op2)

# # Model4 -- 1fc -- one layer, quadratic

class Model4_mnist_gd(Model2_mnist_gd):
            
    def loss(self, labels):
        y = self.y
        self.labels = labels
        self.loss = tf.reduce_mean( (labels-y)**2,name="loss")

class Model4_mnist_bn(Model2_mnist_bn):
    
    def loss(self, labels):
        y = self.y
        self.labels = labels
        self.loss = tf.reduce_mean( (labels-y)**2,name="loss")

#class Model4_mnist_bn_split(Model2_mnist_bn)
class Model4_mnist_bn_split(Model2_mnist_bn_split):
    
    def loss(self, labels):
        y = self.y
        self.labels = labels
        self.loss = tf.reduce_mean( (labels-y)**2,name="loss")

# # Model5 -- 2fc -- one+one layer, quadratic

class Model5_mnist_gd(Model4_mnist_gd):
    
    
    def predict(self, x_image):
        
        x = self.x # not use x_image
        with tf.variable_scope(self.method):

            #flattened = tf.reshape(x, [-1, 28*28])
            flattened = tf.reshape(x, [-1, self.image_size*self.image_size])
            
            hidden = tf.layers.dense(flattened,self.hidden_size,activation=tf.nn.relu, name='fc1')
            
            logits = tf.layers.dense(hidden,10,activation=None, name='fc')

            tf.summary.histogram('logits', logits)

            
        self.logits = logits
        self.y = tf.nn.softmax(logits)

class Model5_mnist_bn(Model4_mnist_bn):
    
    def predict(self, x_image):
        
        x = self.x # not use x_image
        with tf.variable_scope(self.method):

            #flattened = tf.reshape(x, [-1, 28*28])
            flattened = tf.reshape(x, [-1, self.image_size*self.image_size])
            
            hidden1 = tf.layers.dense(flattened,self.hidden_size,activation=None, name='fc1')
            bn1 = tf.layers.batch_normalization(hidden1,training=self.is_training, momentum=0.9, name='bn1')
            layer1 = tf.nn.relu(bn1, name='layer1')
            
            hidden2 = tf.layers.dense(layer1,10,activation=None, name='fc2')
            logits = tf.layers.batch_normalization(hidden2,training=self.is_training, momentum=0.9, name='bn2') 
            
            tf.summary.histogram('logits', logits)
        
        self.logits = logits
        self.y = tf.nn.softmax(logits)
        
        return

class Model5_mnist_bn_split(Model5_mnist_bn):
    
    method = 'bn_split'
    
    def optimizer(self):
        
        #self.training_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.loss)
        
        learning_rate_ab = self.learning_rate_abph
        learning_rate = self.learning_rate
        
        list0 = tf.trainable_variables()
        list2 = tf.trainable_variables(scope='bn_split/bn')
        list1 = list(set(list0)-set(list2))
        print(list1,list2)
        
        with tf.name_scope("train"):
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            #self.training_op = optimizer.minimize(self.loss)
            self.training_op1 = optimizer.minimize(self.loss, var_list=list1)
            
            optimizer2 = tf.train.GradientDescentOptimizer(learning_rate=learning_rate_ab)
            extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(extra_update_ops):
                self.training_op2 = optimizer2.minimize(self.loss, var_list=list2)
            self.training_op = (self.training_op1,self.training_op2)
        return

# # Model_cifar10

class Model_cifar10(Model_mnist):
    image_size = 32
    image_channel = 3
    name = 'cifar10'
    
class Model_cifar10_gd(Model_cifar10):
    method = 'gd'
    
    def predict(self, x_image):
        
        with tf.variable_scope(self.method):

            layer1 = tf.layers.conv2d(x_image, 64, kernel_size=[5,5],strides=[1,1],padding='SAME',
                                      activation=tf.nn.relu, name='layer1')
            pool1 = tf.nn.max_pool(layer1, ksize=[1,3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

            layer2 = tf.layers.conv2d(pool1, 64, kernel_size=[5,5],strides=[1,1],padding='SAME',
                                      activation=tf.nn.relu, name='layer2')
            pool2 = tf.nn.max_pool(layer2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

            flat_shape = pool2.get_shape()[1:4].num_elements() # 8*8*64
            flattened = tf.reshape(pool2, [-1, flat_shape])
            
            fc1 = tf.layers.dense(flattened,512, activation=tf.nn.relu, name='fc1')
            fc2 = tf.layers.dense(fc1,128, activation=tf.nn.relu, name='fc2')
            logits = tf.layers.dense(fc2,10,activation=None, name='fc3')

            tf.summary.histogram('logits', logits)

        self.logits = logits
        self.y = tf.nn.softmax(logits)
    
class Model_cifar10_bn(Model_cifar10):
    method = 'bn'
    
    def predict(self, x_image):
        
        with tf.variable_scope(self.method):

            hidden1 = tf.layers.conv2d(x_image, 64, kernel_size=[5,5],strides=[1,1],padding='SAME',
                                       activation=None, name='hidden1')
            bn1 = tf.layers.batch_normalization(hidden1,training=self.is_training, momentum=0.9, name='bn1')
            layer1 = tf.nn.relu(bn1, name='layer1')
            pool1 = tf.nn.max_pool(bn1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')

            hidden2 = tf.layers.conv2d(pool1, 64, kernel_size=[5,5],strides=[1,1],padding='SAME',
                                       activation=None, name='hidden2')
            bn2 = tf.layers.batch_normalization(hidden2,training=self.is_training, momentum=0.9, name='bn2')
            layer2 = tf.nn.relu(bn1)
            pool2 = tf.nn.max_pool(layer2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

            flat_shape = pool2.get_shape()[1:4].num_elements() # 8*8*64 
            flattened = tf.reshape(pool2, [-1, flat_shape])

            hidden3 = tf.layers.dense(flattened,512,activation=None, name='fc1')
            bn3 = tf.layers.batch_normalization(hidden3,training=self.is_training, momentum=0.9, name='bn3')
            fc1 = tf.nn.relu(bn3)
            
            hidden4 = tf.layers.dense(fc1,128,activation=None, name='fc2')
            bn4 = tf.layers.batch_normalization(hidden4,training=self.is_training, momentum=0.9, name='bn4')
            fc2 = tf.nn.relu(bn4)

            hidden5 = tf.layers.dense(fc2,10,activation=None, name='fc3')
            logits = tf.layers.batch_normalization(hidden5,training=self.is_training, momentum=0.9, name='bn5')  
            
            tf.summary.histogram('logits', logits)
        
        self.logits = logits
        self.y = tf.nn.softmax(logits)

class Model_cifar10_bn_split(Model_cifar10_bn):
    method = 'bn_split'
    
    def optimizer(self):
        
        #self.training_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.loss)
        
        learning_rate_ab = self.learning_rate_abph
        learning_rate = self.learning_rate
        
        list0 = tf.trainable_variables()
        list2 = tf.trainable_variables(scope='bn_split/bn')
        list1 = list(set(list0)-set(list2))
        print(list1,list2)
        
        with tf.name_scope("train"):
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            #self.training_op = optimizer.minimize(self.loss)
            self.training_op1 = optimizer.minimize(self.loss, var_list=list1)
            
            optimizer2 = tf.train.GradientDescentOptimizer(learning_rate=learning_rate_ab)
            extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(extra_update_ops):
                self.training_op2 = optimizer2.minimize(self.loss, var_list=list2)
            self.training_op = (self.training_op1,self.training_op2)

class Model_cifar10_adam(Model_cifar10_gd):
    name = 'adam'
    
    def optimizer(self):
        learning_rate = self.learning_rate
        
        with tf.name_scope("train"):
            optimizer = tf.train.AdamOptimizer(learning_rate)
            #optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            print(extra_update_ops)
            with tf.control_dependencies(extra_update_ops):
                self.training_op = optimizer.minimize(self.loss) 
    

# # Test

class Test(object):

    def test_lr(self, model, dataset,lr_list, n_batch=100,max_step=1000, logdir='../Results/'):
        model.set_tensorboard(logdir)
        value_history = []
        datafiles = []
        for learning_rate in lr_list:
            model.train(dataset=dataset,
                        learning_rate=learning_rate, 
                        n_batch=n_batch,max_step=max_step)
            value_history.append(model.value_history)
            datafiles.append(model.datafile)
        self.lr_list = lr_list
        self.value_history = value_history
        self.datafiles = datafiles
        print(datafiles)
    
    def value_check(self):
        n = len(self.value_history)
        m = max([len(vh) for vh in self.value_history])
        value_history = np.nan * np.ones([n,m,5])
        for ni in range(n):
            mi = len(self.value_history[ni])
            value_history[ni,:mi,:] = np.array(self.value_history[ni])
        self.value_history_np = value_history
        
    def load_value_history(self):
        pass
                
    def plot_lr(self,step=10):
        value_history = self.value_history_np
        
        x = self.lr_list

        #step = 10
        plt.figure(figsize=[20,5])
        plt.subplot(121)
        plt.plot(x,value_history[:,step,1],'b-')
        plt.plot(x,value_history[:,step,3],'r-')
        #plt.xlim([0,10])
        plt.xlabel('learning rate')
        plt.ylabel('loss at step=%d'%step);        
        plt.legend(('train','test'))

        plt.subplot(122)
        plt.semilogx(x,value_history[:,step,2],'b-')
        plt.plot(x,value_history[:,step,4],'r-')
        #plt.xlim([0,10])
        plt.xlabel('learning rate')
        plt.ylabel('accuracy at step=%d'%step);
        plt.legend(('train','test'))
#import numpy as np
import scipy as scipy

#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

def imresize(x,size): # resize one image such as 28*28 --> 20*20
    xr = scipy.misc.imresize(x,size)
    #return np.array(xr,dtype='float32')
    return np.array(xr,dtype='float32') / 255.0
def imresize_mnist_batch(xb,size): # resize image batch
    n_batch = len(xb)
    size0 = [28,28]
    xbr = np.zeros([n_batch,size[0]*size[1]])
    for i in range(n_batch):
        x = xb[i].reshape(size0)
        xr = imresize(x,size)
        xbr[i,:] = xr.reshape([1,size[0]*size[1]])
    return xbr

class mnist_resized():
    def __init__(self,mnist, trainORtest,size,**args):
        self.trainORtest = trainORtest
        self.size = size
        func_next_batch = [mnist.train.next_batch, mnist.test.next_batch]
        self.func = func_next_batch[trainORtest]
        return
    def next_batch(self,n_batch):
        x,y = self.func(n_batch)
        xr = imresize_mnist_batch(x,self.size)
        #print('xr',xr.shape)
        return xr,y
    
class dataset_mnist_resized():
    def __init__(self, mnist, size, **args):
        self.train = mnist_resized(mnist, 0,size)
        self.test = mnist_resized(mnist, 1,size)
        return
    
#dataset2 = dataset_mnist_resized(mnist, [22,22])
#xb,yb = dataset2.train.next_batch(3)
#xt,yt = dataset2.test.next_batch(4)
#xb.shape,yb.shape, xt.shape,yt.shape