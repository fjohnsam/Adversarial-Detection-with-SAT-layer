# Name : David Johnson Ekka
# Roll no : 19CS60R15
# Assignment no :10

from sklearn.model_selection import train_test_split as split
import tensorflow as tf
from tensorflow.keras.datasets import mnist

import pickle

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

if tf.__version__ < "2.0.0":
    print("Requires Tensorflow 2.0.0 or later")
    exit(0)


class neural_net:
    def __init__(self, **args):
        # Seed
        self.seed = args["seed"]
        tf.random.set_seed(self.seed)

        # Paths
        self.weight_file = args["weight_file"]
        self.data_path = args["data_path"]

        # Fashion MNIST parameters
        self.n_classes = 10
        self.n_features = 784

        # Network Parameters
        self.n_hidden = args["n_hidden"]
        self.weights = None  # Dict containing weights and biases
        self.train_data = None
        # self.weights = None

        # Convolution parameters
        self.conv1_filters = 32 # number of filters for 1st conv layer.
        self.conv2_filters = 64 # number of filters for 2nd conv layer.
        self.fc1_units = 1024 # number of neurons for 1st fully-connected layer.
        self.num_classes = 10

        # Training Parameters
        self.learning_rate = args["learning_rate"]
        self.batch_size = 64
        self.shuffle_size = 60000
        self.display_step = 5
        self.weightsatience = args["patience"]
        self.reg_parm = 0.01
        self.train_steps = 200

        # Training and test data
        self.x_test = None
        self.x_train = None
        self.x_val = None
        self.y_test = None
        self.y_train = None
        self.y_val = None

        # Optimizer
        self.optimizer = tf.optimizers.SGD(self.learning_rate)

    # Save network weights to file
    def save_weight(self):
        out_file = open(self.weight_file, "wb")
        pickle.dump(self.weights, out_file)
        out_file.close()

    def load_weight(self):
        infile = open(self.weight_file, "rb")
        self.weights = pickle.load(infile)
        infile.close()

    # Activation functions
    # Function for softmax activation
    def activation_softmax(self, val):
        max = tf.transpose([tf.math.reduce_max(val, axis=1)])
        val = tf.subtract(val, max)
        sum = tf.transpose([tf.math.reduce_sum(tf.exp(val), axis=1)])
        return tf.exp(val) / sum

    # Function for relu activation
    def activation_relu(self, val):
        return tf.maximum(val, 0.00)

    # Create some wrappers for simplicity.
    def conv2d(self,x, W, b, strides=1):
        # Conv2D wrapper, with bias and relu activation.
        x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
        x = tf.nn.bias_add(x, b)
        return tf.nn.relu(x)

    def maxpool2d(self,x, k=2):
        # MaxPool2D wrapper.
        return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

    # Layer definition
    def layer(self, X, layer_no):
        return tf.add(tf.matmul(X, self.weights["w" + layer_no]), self.weights["b" + layer_no])

    # Defining the model of the neural network
    def network(self, X, layer=None):
        # Input shape: [-1, 28, 28, 1]. A batch of 28x28x1 (grayscale) images.
        x = tf.reshape(X, [-1, 28, 28, 1])

        # Convolution Layer. Output shape: [-1, 28, 28, 32].
        conv1 = self.conv2d(x, self.weights['wc1'], self.weights['bc1'])

        # Max Pooling (down-sampling). Output shape: [-1, 14, 14, 32].
        conv1 = self.maxpool2d(conv1, k=2)

        # Convolution Layer. Output shape: [-1, 14, 14, 64].
        conv2 = self.conv2d(conv1, self.weights['wc2'], self.weights['bc2'])

        # Max Pooling (down-sampling). Output shape: [-1, 7, 7, 64].
        conv2 = self.maxpool2d(conv2, k=2)

        # Reshape conv2 output to fit fully connected layer input, Output shape: [-1, 7*7*64].
        fc1 = tf.reshape(conv2, [-1, self.weights['wd1'].get_shape().as_list()[0]])

        # Fully connected layer, Output shape: [-1, 1024].
        fc1 = tf.add(tf.matmul(fc1, self.weights['wd1']), self.weights['bd1'])
        # Apply ReLU to fc1 output for non-linearity.
        fc1 = tf.nn.relu(fc1)

        # Fully connected layer, Output shape: [-1, 10].
        out = tf.add(tf.matmul(fc1, self.weights['out']), self.weights['out_b'])
        # Apply softmax to normalize the logits to a probability distribution.
        return tf.nn.softmax(out)

    # Loading Fashion MNIST data from path
    # and dividing train data into train and validation sets
    def load_data(self, val_set_size=0.20):

        (X, Y), (x_test, y_test) = mnist.load_data()


        # Splitting Data into train and validation sets
        # x_train, x_val, y_train, y_val = split(X, Y, test_size=val_set_size, shuffle=True)
        x_train, x_val, y_train, y_val = split(X, Y, test_size=val_set_size, random_state=self.seed, shuffle=True)

        self.y_train = y_train
        self.y_test = y_test
        self.y_val = y_val

        # Flattening Data
        x_train = tf.reshape(tf.Variable(x_train, dtype=tf.float32), [len(x_train), 784])
        x_val = tf.reshape(tf.Variable(x_val, dtype=tf.float32), [len(x_val), 784])
        x_test = tf.reshape(tf.Variable(x_test, dtype=tf.float32), [len(x_test), 784])

        # Normalizing Data
        self.x_train = x_train / 255
        self.x_test = x_test / 255
        self.x_val = x_val / 255

    # Create training batches and initialize network weights
    def net_initialize(self):
        # Shuffle and batch data.
        self.train_data = tf.data.Dataset.from_tensor_slices((self.x_train, self.y_train))
        self.train_data = self.train_data.repeat().shuffle(len(self.x_train), reshuffle_each_iteration=True).batch(self.batch_size).prefetch(1)

        # A random value generator to initialize weights.
        random_normal = tf.initializers.RandomNormal()

        # Layer weights and bias
        # Weights start with w
        # Bias terms start with b
        self.weights = {
            # Conv Layer 1: 5x5 conv, 1 input, 32 filters (MNIST has 1 color channel only).
            'wc1': tf.Variable(random_normal([5, 5, 1, self.conv1_filters])),
            # Conv Layer 2: 5x5 conv, 32 inputs, 64 filters.
            'wc2': tf.Variable(random_normal([5, 5, self.conv1_filters, self.conv2_filters])),
            # FC Layer 1: 7*7*64 inputs, 1024 units.
            'wd1': tf.Variable(random_normal([7*7*64, self.fc1_units])),
            # FC Out Layer: 1024 inputs, 10 units (total number of classes)
            'out': tf.Variable(random_normal([self.fc1_units, self.num_classes])),
            'bc1': tf.Variable(tf.zeros([self.conv1_filters])),
            'bc2': tf.Variable(tf.zeros([self.conv2_filters])),
            'bd1': tf.Variable(tf.zeros([self.fc1_units])),
            'out_b': tf.Variable(tf.zeros([self.num_classes]))
        }

    # L2 regualarizer
    def l2_regularizer(self):
        sum = 0.0
        # Getting list of weights
        weights = [x for x in self.weights.keys() if x.startswith("w")]
        for index in weights:
            sum += self.reg_parm * tf.reduce_sum(tf.square(self.weights[index]))
        return sum

    # Cross-entropy loss function
    def cross_entropy_loss(self, Y, Y_T):
        Y_T = tf.one_hot(Y_T, depth=self.n_classes)
        Y = tf.clip_by_value(Y, 1e-9, 1.0)
        out = tf.reduce_mean(-tf.reduce_sum(Y_T * tf.math.log(Y)))
        return out

    # Function for calculating Accuracy
    def accuracy(self, Y, Y_T):
        y_t = tf.cast(Y_T, tf.int64)
        c_prediction = tf.cast(tf.equal(tf.argmax(Y, 1), y_t), tf.float32)
        mean = tf.reduce_mean(c_prediction)
        return mean

    def run_optimization(self, X, Y):
        with tf.GradientTape() as g:
            prediction = self.network(X)
            loss = self.cross_entropy_loss(prediction, Y) + self.l2_regularizer()

        # Calculating gradients
        gradients = g.gradient(loss, self.weights)

        # Updating weights
        for index in gradients.keys():
            self.optimizer.apply_gradients(zip([gradients[index]], [self.weights[index]]))

        return prediction, loss

    def train_network(self):
        count = 0  # Counter for patience
        best_acc = tf.Variable(0, dtype=tf.float32)  # Variable storing best accuracy on validation set
        prev_acc = tf.Variable(0, dtype=tf.float32)
        for counter, (X, Y) in enumerate(self.train_data.take(self.train_steps),1):
            prediction, loss = self.run_optimization(X, Y)
            accuracy = self.accuracy(prediction, Y)

            if counter % self.display_step == 0:
                print(f"Epoch : {counter:>4} | Accuracy: {accuracy:<.3f} | Loss: {loss:>4.3f} ")


if __name__ == "__main__":
    obj = neural_net(
        weight_file="Weights/weight_dict",
        data_path="../data/fashion",
        seed=23,
        n_hidden=512,
        patience=36,
        learning_rate=0.001,
    )
    obj.load_data()

    obj.patience = 20
    obj.n_hidden = 512
    obj.shuffle_size = 60000
    obj.display_step = 10
    obj.net_initialize()
    obj.train_network()

    # Testing on test data
    pred = obj.network(obj.x_test)
    accu = obj.accuracy(pred, obj.y_test)
    print(f"Test accuracy : {accu:<.3f}")
