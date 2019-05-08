import pickle
import numpy as np
#import matplotlib.pyplot as plt

class Cifar10(object):
    
    def __init__(self,dirpath='./cifar10_data', one_hot=True,normalize=False):
        
        train_data, train_label, test_data, test_label = self.load_dataset(dirpath, one_hot, normalize)
        assert len(train_data) == len(train_label)
        assert len(test_data) == len(test_label)
        self.train = Dataset(train_data, train_label)
        self.test = Dataset(test_data, test_label)
        
    def load_dataset(self,dirpath='./cifar10_data', one_hot=True,normalize=False):
        x, y = [], []
        # take data from the data batch
        for i in range(5):
            path = '%s/data_batch_%d' % (dirpath, i+1)        
            with open(path, 'rb') as f:
                batch = pickle.load(f,encoding='latin1')
            x.append(batch['data'])
            y.append(batch['labels'])
        x = np.concatenate(x) /np.float32(255) # making value 0 to 1
        x_train = x.reshape((-1, 32*32*3))
        y_train = np.concatenate(y).astype(np.int32) # making labels as int

        # load test set
        path = '%s/test_batch' % dirpath
        with open(path, 'rb') as f:
            batch = pickle.load(f,encoding='latin1')
        x_test = batch['data'] /np.float32(255)
        x_test = x_test.reshape((-1, 32*32*3))
        y_test = np.array(batch['labels'], dtype=np.int32)


        if one_hot:
            y = y_train
            y_train = np.zeros((50000,10),dtype = np.float32)        
            for i in range(50000):
                a = y[i]
                y_train[i,a] = 1

            y = y_test
            y_test = np.zeros((10000,10),dtype = np.int32)
            for i in range(10000):
                a = y[i]
                y_test[i,a] = 1

        if normalize:        
            # normalize to zero mean and unity variance
            offset = np.mean(x_train, 0)
            scale = np.std(x_train, 0).clip(min=1)
            x_train = (x_train - offset) / scale
            x_test = (x_test - offset) / scale

        return x_train,y_train,x_test,y_test
    
    def load_dataset_rgb(self,dirpath='./cifar10_data', one_hot=True,normalize=False):
        x, y = [], []
        # take data from the data batch
        for i in range(5):
            path = '%s/data_batch_%d' % (dirpath, i+1)        
            with open(path, 'rb') as f:
                batch = pickle.load(f,encoding='latin1')
            x.append(batch['data'])
            y.append(batch['labels'])
        x = np.concatenate(x) /np.float32(255) # making value 0 to 1
        x = np.dstack((x[:, :1024], x[:, 1024:2048], x[:, 2048:])) # RGB
        x_train = x.reshape((x.shape[0], 32, 32, 3))
        y_train = np.concatenate(y).astype(np.int32) # making labels as int

        # load test set
        path = '%s/test_batch' % dirpath
        with open(path, 'rb') as f:
            batch = pickle.load(f,encoding='latin1')
        x_test = batch['data'] /np.float32(255)
        x_test = np.dstack((x_test[:, :1024], x_test[:, 1024:2048], x_test[:, 2048:]))
        x_test = x_test.reshape((x_test.shape[0], 32, 32, 3))
        y_test = np.array(batch['labels'], dtype=np.int32)


        if one_hot:
            y = y_train
            y_train = np.zeros((50000,10),dtype = np.float32)        
            for i in range(50000):
                a = y[i]
                y_train[i,a] = 1

            y = y_test
            y_test = np.zeros((10000,10),dtype = np.int32)
            for i in range(10000):
                a = y[i]
                y_test[i,a] = 1

        if normalize:        
            # normalize to zero mean and unity variance
            offset = np.mean(x_train, 0)
            scale = np.std(x_train, 0).clip(min=1)
            x_train = (x_train - offset) / scale
            x_test = (x_test - offset) / scale

        return x_train,y_train,x_test,y_test
        

class Dataset(object):

    def __init__(self, data, label, shuffle=True):
        assert len(data.shape) > 1 and len(label.shape) > 1
        assert data.shape[0] == label.shape[0]
        self._num_examples = data.shape[0]
        self._data = data
        self._label = label
        self._epoch_completed = 0
        self._index_in_epoch = 0
        self.shuffle = shuffle

    @property
    def data(self):
        return self._data

    @property
    def label(self):
        return self._label

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def data_shape(self):
        return self._data.shape[1:]

    @property
    def label_shape(self):
        return self._label.shape[1:]

    def next_batch(self, batch_size):
        if self.shuffle:
            return self.random_next_batch(batch_size)
        else:
            return self._next_batch(batch_size)
            
        
    def _next_batch(self, batch_size):
    
        train_size = self._num_examples
        start = self._index_in_epoch
        end = self._index_in_epoch + batch_size \
            if self._index_in_epoch+batch_size < train_size else train_size
        if end > train_size:
            self._epoch_completed += 1
            # self._data = shuffle(self._data)
            # self._label = shuffle(self._label)
        x_batch, y_batch = \
            self._data[start:end], \
            self._label[start:end]
        self._index_in_epoch = end if end < train_size else 0
        return x_batch, y_batch

    def random_next_batch(self, batch_size, weights=None):
        if weights is not None:
            assert np.all(weights >= 0)
            assert len(weights) == self._num_examples
            weights = weights / np.sum(weights)
        indices = np.arange(self._num_examples)
        batch_indices = np.random.choice(
            indices, size=(batch_size,), p=weights)
        x_batch = self._data[batch_indices]
        y_batch = self._label[batch_indices]
        return x_batch, y_batch
        