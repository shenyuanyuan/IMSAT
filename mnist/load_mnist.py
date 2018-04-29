from sklearn.datasets import fetch_mldata
import numpy as np

train_num = 60000
test_num = 10000

class Dataset():
	def __init__(self, train, test):
		self.images_train, self.labels_train = train
		self.images_test, self.labels_test = test
		self.dataset_indices = np.arange(0, len(self.images_train))
		self.shuffle()

	def sample_minibatch(self, batchsize):
		batch_indices = self.dataset_indices[:batchsize]
		x_batch = self.images_train[batch_indices]
		y_batch = self.labels_train[batch_indices]
		self.dataset_indices = np.roll(self.dataset_indices, batchsize)
		return x_batch, y_batch

	def shuffle(self):
		np.random.shuffle(self.dataset_indices)

def load_mnist_whole(scale, PATH = '.'):
	print('fetch MNIST dataset')
	mnist = fetch_mldata('MNIST original', data_home=PATH)
	data = mnist.data.astype(np.float32)*scale
	target = mnist.target.astype(np.int32)
	train_data = data[:train_num]
	train_labels = target[:train_num]
	test_data = data[train_num:]
    test_labels = target[train_num:]
    dataset = Dataset(train=(train_data, train_labels), test=(test_data, test_labels))

    print("load mnist done")
    return dataset
