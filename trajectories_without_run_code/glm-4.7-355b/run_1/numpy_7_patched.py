# --- [CELL 0]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 1}
import numpy as np


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def cross_entropy(x):
    return -np.log(x)


def regularized_cross_entropy(layers, lam, x):
    loss = cross_entropy(x)
    for layer in layers:
        loss += lam * (np.linalg.norm(layer.get_weights()) ** 2)
    return loss


def leakyReLU(x, alpha=0.001):
    return x * alpha if x < 0 else x


def leakyReLU_derivative(x, alpha=0.01):
    return alpha if x < 0 else 1


def lr_schedule(learning_rate, iteration): # updated code , this will fix the error 
    if iteration == 0:
        return learning_rate
    if (iteration >= 0) and (iteration <= 10000):
        return learning_rate
    if iteration > 10000:
        return learning_rate * 0.1
    if iteration > 30000:
        return learning_rate * 0.1

class Convolutional:                                        # convolution layer using 3x3 filters

    def __init__(self, name, num_filters=16, stride=1, size=3, activation=None):
        self.name = name
        self.filters = np.random.randn(num_filters, 3, 3) * 0.1
        self.stride = stride
        self.size = size
        self.activation = activation
        self.last_input = None
        self.leakyReLU = np.vectorize(leakyReLU)
        self.leakyReLU_derivative = np.vectorize(leakyReLU_derivative)

    def forward(self, image):
        self.last_input = image                             # keep track of last input for later backward propagation

        input_dimension = image.shape[1]                                                # input dimension
        output_dimension = int((input_dimension - self.size) / self.stride) + 1         # output dimension

        out = np.zeros((self.filters.shape[0], output_dimension, output_dimension))     # create the matrix to hold the
                                                                                        # values of the convolution

        for f in range(self.filters.shape[0]):              # convolve each filter over the image,
            tmp_y = out_y = 0                               # moving it vertically first and then horizontally
            while tmp_y + self.size <= input_dimension:
                tmp_x = out_x = 0
                while tmp_x + self.size <= input_dimension:
                    patch = image[:, tmp_y:tmp_y + self.size, tmp_x:tmp_x + self.size]
                    out[f, out_y, out_x] += np.sum(self.filters[f] * patch)
                    tmp_x += self.stride
                    out_x += 1
                tmp_y += self.stride
                out_y += 1
        if self.activation == 'relu':                       # apply ReLU activation function
            self.leakyReLU(out)
        return out

    def backward(self, din, learn_rate=0.005):
        input_dimension = self.last_input.shape[1]          # input dimension

        if self.activation == 'relu':                       # back propagate through ReLU
           self.leakyReLU_derivative(din)

        dout = np.zeros(self.last_input.shape)              # loss gradient of the input to the convolution operation
        dfilt = np.zeros(self.filters.shape)                # loss gradient of filter

        for f in range(self.filters.shape[0]):              # loop through all filters
            tmp_y = out_y = 0
            while tmp_y + self.size <= input_dimension:
                tmp_x = out_x = 0
                while tmp_x + self.size <= input_dimension:
                    patch = self.last_input[:, tmp_y:tmp_y + self.size, tmp_x:tmp_x + self.size]
                    dfilt[f] += np.sum(din[f, out_y, out_x] * patch, axis=0)
                    dout[:, tmp_y:tmp_y + self.size, tmp_x:tmp_x + self.size] += din[f, out_y, out_x] * self.filters[f]
                    tmp_x += self.stride
                    out_x += 1
                tmp_y += self.stride
                out_y += 1
        self.filters -= learn_rate * dfilt                  # update filters using SGD
        return dout                                         # return the loss gradient for this layer's inputs

    def get_weights(self):
        return np.reshape(self.filters, -1)


class Pooling:                                              # max pooling layer using pool size equal to 2
    def __init__(self, name, stride=2, size=2):
        self.name = name
        self.last_input = None
        self.stride = stride
        self.size = size

    def forward(self, image):
        self.last_input = image                             # keep track of last input for later backward propagation

        num_channels, h_prev, w_prev = image.shape
        h = int((h_prev - self.size) / self.stride) + 1     # compute output dimensions after the max pooling
        w = int((w_prev - self.size) / self.stride) + 1

        downsampled = np.zeros((num_channels, h, w))        # hold the values of the max pooling

        for i in range(num_channels):                       # slide the window over every part of the image and
            curr_y = out_y = 0                              # take the maximum value at each step
            while curr_y + self.size <= h_prev:             # slide the max pooling window vertically across the image
                curr_x = out_x = 0
                while curr_x + self.size <= w_prev:         # slide the max pooling window horizontally across the image
                    patch = image[i, curr_y:curr_y + self.size, curr_x:curr_x + self.size]
                    downsampled[i, out_y, out_x] = np.max(patch)       # choose the maximum value within the window
                    curr_x += self.stride                              # at each step and store it to the output matrix
                    out_x += 1
                curr_y += self.stride
                out_y += 1

        return downsampled

    def backward(self, din, learning_rate):
        num_channels, orig_dim, *_ = self.last_input.shape      # gradients are passed through the indices of greatest
                                                                # value in the original pooling during the forward step

        dout = np.zeros(self.last_input.shape)                  # initialize derivative

        for c in range(num_channels):
            tmp_y = out_y = 0
            while tmp_y + self.size <= orig_dim:
                tmp_x = out_x = 0
                while tmp_x + self.size <= orig_dim:
                    patch = self.last_input[c, tmp_y:tmp_y + self.size, tmp_x:tmp_x + self.size]    # obtain index of largest
                    (x, y) = np.unravel_index(np.nanargmax(patch), patch.shape)                     # value in patch
                    dout[c, tmp_y + x, tmp_x + y] += din[c, out_y, out_x]
                    tmp_x += self.stride
                    out_x += 1
                tmp_y += self.stride
                out_y += 1

        return dout

    def get_weights(self):                          # pooling layers have no weights
        return 0


class FullyConnected:                               # fully-connected layer
    def __init__(self, name, nodes1, nodes2, activation):
        self.name = name
        self.weights = np.random.randn(nodes1, nodes2) * 0.1
        self.biases = np.zeros(nodes2)
        self.activation = activation
        self.last_input_shape = None
        self.last_input = None
        self.last_output = None
        self.leakyReLU = np.vectorize(leakyReLU)
        self.leakyReLU_derivative = np.vectorize(leakyReLU_derivative)

    def forward(self, input):
        self.last_input_shape = input.shape         # keep track of last input shape before flattening
                                                    # for later backward propagation

        input = input.flatten()                                 # flatten input

        output = np.dot(input, self.weights) + self.biases      # forward propagate

        if self.activation == 'relu':                           # apply ReLU activation function
            self.leakyReLU(output)

        self.last_input = input                     # keep track of last input and output for later backward propagation
        self.last_output = output

        return output

    def backward(self, din, learning_rate=0.005):
        if self.activation == 'relu':                             # back propagate through ReLU
           self.leakyReLU_derivative(din)

        self.last_input = np.expand_dims(self.last_input, axis=1)
        din = np.expand_dims(din, axis=1)

        dw = np.dot(self.last_input, np.transpose(din))           # loss gradient of final dense layer weights
        db = np.sum(din, axis=1).reshape(self.biases.shape)       # loss gradient of final dense layer biases

        self.weights -= learning_rate * dw                        # update weights and biases
        self.biases -= learning_rate * db

        dout = np.dot(self.weights, din)
        return dout.reshape(self.last_input_shape)

    def get_weights(self):
        return np.reshape(self.weights, -1)


class Dense:                                        # dense layer with softmax activation
    def __init__(self, name, nodes, num_classes):
        self.name = name
        self.weights = np.random.randn(nodes, num_classes) * 0.1
        self.biases = np.zeros(num_classes)
        self.last_input_shape = None
        self.last_input = None
        self.last_output = None

    def forward(self, input):
        self.last_input_shape = input.shape         # keep track of last input shape before flattening
                                                    # for later backward propagation

        input = input.flatten()                                 # flatten input

        output = np.dot(input, self.weights) + self.biases      # forward propagate

        self.last_input = input                     # keep track of last input and output for later backward propagation
        self.last_output = output

        return softmax(output)

    def backward(self, din, learning_rate=0.005):
        for i, gradient in enumerate(din):
            if gradient == 0:                   # the derivative of the loss with respect to the output is nonzero
                continue                        # only for the correct class, so skip if the gradient is zero

            t_exp = np.exp(self.last_output)                      # gradient of dout[i] with respect to output
            dout_dt = -t_exp[i] * t_exp / (np.sum(t_exp) ** 2)
            dout_dt[i] = t_exp[i] * (np.sum(t_exp) - t_exp[i]) / (np.sum(t_exp) ** 2)

            dt = gradient * dout_dt                               # gradient of loss with respect to output

            dout = self.weights @ dt                              # gradient of loss with respect to input

            # update weights and biases
            self.weights -= learning_rate * (np.transpose(self.last_input[np.newaxis]) @ dt[np.newaxis])
            self.biases -= learning_rate * dt

            return dout.reshape(self.last_input_shape)            # return the loss gradient for this layer's inputs

    def get_weights(self):
        return np.reshape(self.weights, -1)

#%%
# --- [CELL 1]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 2}
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import seaborn as sns
import idx2numpy
import numpy as np
from six.moves import cPickle
import platform
import cv2
sns.set(color_codes=True)

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)          # suppress messages from TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def load_mnist():
    X_train = idx2numpy.convert_from_file('MNIST_data/train-images-idx3-ubyte')
    train_labels = idx2numpy.convert_from_file('MNIST_data/train-labels-idx1-ubyte')
    X_test = idx2numpy.convert_from_file('MNIST_data/t10k-images-idx3-ubyte')
    test_labels = idx2numpy.convert_from_file('MNIST_data/t10k-labels-idx1-ubyte')

    train_images = []                                                   # reshape train images so that the training set
    for i in range(X_train.shape[0]):                                   # is of shape (60000, 1, 28, 28)
        train_images.append(np.expand_dims(X_train[i], axis=0))
    train_images = np.array(train_images)

    test_images = []                                                    # reshape test images so that the test set
    for i in range(X_test.shape[0]):                                    # is of shape (10000, 1, 28, 28)
        test_images.append(np.expand_dims(X_test[i], axis=0))
    test_images = np.array(test_images)

    indices = np.random.permutation(train_images.shape[0])              # permute and split training data in
    training_idx, validation_idx = indices[:55000], indices[55000:]     # training and validation sets
    train_images, validation_images = train_images[training_idx, :], train_images[validation_idx, :]
    train_labels, validation_labels = train_labels[training_idx], train_labels[validation_idx]

    return {
        'train_images': train_images,
        'train_labels': train_labels,
        'validation_images': validation_images,
        'validation_labels': validation_labels,
        'test_images': test_images,
        'test_labels': test_labels
    }


def load_pickle(f):
    version = platform.python_version_tuple()
    if version[0] == '2':
        return cPickle.load(f)
    elif version[0] == '3':
        return cPickle.load(f, encoding='latin1')
    raise ValueError("invalid python version: {}".format(version))


from keras.datasets import cifar10
import numpy as np

def load_cifar():
    # Load the CIFAR-10 training and test data
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    # Permute and split training data into training and validation sets
    indices = np.random.permutation(X_train.shape[0])
    training_idx, validation_idx = indices[:49000], indices[49000:]
    X_train, X_val = X_train[training_idx, :], X_train[validation_idx, :]
    y_train, y_val = y_train[training_idx], y_train[validation_idx]

    return {
        'train_images': X_train,
        'train_labels': y_train,
        'validation_images': X_val,
        'validation_labels': y_val,
        'test_images': X_test,
        'test_labels': y_test
    }

def minmax_normalize(x):
    min_val = np.min(x)
    max_val = np.max(x)
    x = (x - min_val) / (max_val - min_val)
    return x


def preprocess(dataset):
    dataset['train_images'] = np.array([minmax_normalize(x) for x in dataset['train_images']])
    dataset['validation_images'] = np.array([minmax_normalize(x) for x in dataset['validation_images']])
    dataset['test_images'] = np.array([minmax_normalize(x) for x in dataset['test_images']])
    return dataset


def plot_accuracy_curve(accuracy_history, val_accuracy_history):
    plt.plot(accuracy_history, 'b', linewidth=3.0, label='Training accuracy')
    plt.plot(val_accuracy_history, 'r', linewidth=3.0, label='Validation accuracy')
    plt.xlabel('Iteration', fontsize=16)
    plt.ylabel('Accuracy rate', fontsize=16)
    plt.legend()
    plt.title('Training Accuracy', fontsize=16)
    plt.savefig('training_accuracy.png')
    plt.show()


def plot_learning_curve(loss_history):
    plt.plot(loss_history, 'b', linewidth=3.0, label='Cross entropy')
    plt.xlabel('Iteration', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.legend()
    plt.title('Learning Curve', fontsize=16)
    plt.savefig('learning_curve.png')
    plt.show()


def plot_sample(image, true_label, predicted_label):
    plt.imshow(image)
    if true_label and predicted_label is not None:
        if type(true_label) == 'int':
            plt.title('True label: %d, Predicted Label: %d' % (true_label, predicted_label))
        else:
            plt.title('True label: %s, Predicted Label: %s' % (true_label, predicted_label))
    plt.show()


def plot_histogram(layer_name, layer_weights):
    plt.hist(layer_weights)
    plt.title('Histogram of ' + str(layer_name))
    plt.xlabel('Value')
    plt.ylabel('Number')
    plt.show()


def to_gray(image_name):
    image = cv2.imread(image_name + '.png')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Gray image', image)
    cv2.imwrite(image_name + '.png', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#%%
# --- [CELL 2]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 3}
import numpy as np
import time


class Network:
    def __init__(self):
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def build_model(self, dataset_name):
        if dataset_name == 'mnist':
            self.add_layer(Convolutional(name='conv1', num_filters=8, stride=2, size=3, activation='relu'))
            self.add_layer(Convolutional(name='conv2', num_filters=8, stride=2, size=3, activation='relu'))
            self.add_layer(Dense(name='dense', nodes=8 * 6 * 6, num_classes=10))
        else:
            self.add_layer(Convolutional(name='conv1', num_filters=32, stride=1, size=3, activation='relu'))
            self.add_layer(Convolutional(name='conv2', num_filters=32, stride=1, size=3, activation='relu'))
            self.add_layer(Pooling(name='pool1', stride=2, size=2))
            self.add_layer(Convolutional(name='conv3', num_filters=64, stride=1, size=3, activation='relu'))
            self.add_layer(Convolutional(name='conv4', num_filters=64, stride=1, size=3, activation='relu'))
            self.add_layer(Pooling(name='pool2', stride=2, size=2))
            self.add_layer(FullyConnected(name='fullyconnected', nodes1=64 * 5 * 5, nodes2=256, activation='relu'))
            self.add_layer(Dense(name='dense', nodes=256, num_classes=10))

    def forward(self, image, plot_feature_maps):                # forward propagate
        for layer in self.layers:
            if plot_feature_maps:
                image = (image * 255)[0, :, :]
                plot_sample(image, None, None)
            image = layer.forward(image)
        return image

    def backward(self, gradient, learning_rate):                # backward propagate
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient, learning_rate)

    def train(self, dataset, num_epochs, learning_rate, validate, regularization, plot_weights, verbose):
        history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
        for epoch in range(1, num_epochs + 1):
            print('\n--- Epoch {0} ---'.format(epoch))
            loss, tmp_loss, num_corr = 0, 0, 0
            initial_time = time.time()
            for i in range(len(dataset['train_images'])):
                if i % 100 == 99:
                    accuracy = (num_corr / (i + 1)) * 100       # compute training accuracy and loss up to iteration i
                    loss = tmp_loss / (i + 1)

                    history['loss'].append(loss)                # update history
                    history['accuracy'].append(accuracy)

                    if validate:
                        indices = np.random.permutation(dataset['validation_images'].shape[0])
                        val_loss, val_accuracy = self.evaluate(
                            dataset['validation_images'][indices, :],
                            dataset['validation_labels'][indices],
                            regularization,
                            plot_correct=0,
                            plot_missclassified=0,
                            plot_feature_maps=0,
                            verbose=0
                        )
                        history['val_loss'].append(val_loss)
                        history['val_accuracy'].append(val_accuracy)

                        if verbose:
                            print('[Step %05d]: Loss %02.3f | Accuracy: %02.3f | Time: %02.2f seconds | '
                                  'Validation Loss %02.3f | Validation Accuracy: %02.3f' %
                                  (i + 1, loss, accuracy, time.time() - initial_time, val_loss, val_accuracy))
                    elif verbose:
                        print('[Step %05d]: Loss %02.3f | Accuracy: %02.3f | Time: %02.2f seconds' %
                              (i + 1, loss, accuracy, time.time() - initial_time))

                    # restart time
                    initial_time = time.time()

                image = dataset['train_images'][i]
                label = dataset['train_labels'][i]

                tmp_output = self.forward(image, plot_feature_maps=0)       # forward propagation

                # compute (regularized) cross-entropy and update loss
                tmp_loss += regularized_cross_entropy(self.layers, regularization, tmp_output[label])

                if np.argmax(tmp_output) == label:                          # update accuracy
                    num_corr += 1

                gradient = np.zeros(10)                                     # compute initial gradient
                gradient[label] = -1 / tmp_output[label] + np.sum(
                    [2 * regularization * np.sum(np.absolute(layer.get_weights())) for layer in self.layers])

                learning_rate = lr_schedule(learning_rate, iteration=i)     # learning rate decay

                self.backward(gradient, learning_rate)                      # backward propagation

        if verbose:
            print('Train Loss: %02.3f' % (history['loss'][-1]))
            print('Train Accuracy: %02.3f' % (history['accuracy'][-1]))
            plot_learning_curve(history['loss'])
            plot_accuracy_curve(history['accuracy'], history['val_accuracy'])

        if plot_weights:
            for layer in self.layers:
                if 'pool' not in layer.name:
                    plot_histogram(layer.name, layer.get_weights())

    def evaluate(self, X, y, regularization, plot_correct, plot_missclassified, plot_feature_maps, verbose):
        loss, num_correct = 0, 0
        for i in range(len(X)):
            tmp_output = self.forward(X[i], plot_feature_maps)              # forward propagation

            # compute cross-entropy update loss
            loss += regularized_cross_entropy(self.layers, regularization, tmp_output[y[i]])

            prediction = np.argmax(tmp_output)                              # update accuracy
            if prediction == y[i]:
                num_correct += 1
                if plot_correct:                                            # plot correctly classified digit
                    image = (X[i] * 255)[0, :, :]
                    plot_sample(image, y[i], prediction)
                    plot_correct = 1
            else:
                if plot_missclassified:                                     # plot missclassified digit
                    image = (X[i] * 255)[0, :, :]
                    plot_sample(image, y[i], prediction)
                    plot_missclassified = 1

        test_size = len(X)
        accuracy = (num_correct / test_size) * 100
        loss = loss / test_size
        if verbose:
            print('Test Loss: %02.3f' % loss)
            print('Test Accuracy: %02.3f' % accuracy)
        return loss, accuracy

#%%
# --- [CELL 3]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 4}
dataset = load_cifar()

# Access the data and labels
X_train = dataset['train_images']
y_train = dataset['train_labels']
X_val = dataset['validation_images']
y_val = dataset['validation_labels']
X_test = dataset['test_images']
y_test = dataset['test_labels']

# Print the shape of the data and labels
print("Training data shape:", X_train.shape)
print("Training labels shape:", y_train.shape)
print("Validation data shape:", X_val.shape)
print("Validation labels shape:", y_val.shape)
print("Test data shape:", X_test.shape)
print("Test labels shape:", y_test.shape)

#%%
# --- [CELL 4]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 5}
# === BEFORE (original) ===
# from keras.regularizers import l2
# from keras.optimizers import SGD
# from keras.models import Sequential
# from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
# from keras.utils import to_categorical
# from keras.callbacks import Callback
# from keras import Model
# from keras.models import load_model
# import numpy as np
# import os
# import matplotlib.pyplot as plt
# 
# 
# class History(Callback):
#     def __init__(self, model, validation_images, validation_labels):
#         self.model_ = model
#         self.validation_images = validation_images
#         self.validation_labels = validation_labels
#         self.accuracy = [0]
#         self.loss = [5]
#         self.val_accuracy = [0]
#         self.val_loss = [5]
# 
#     def on_batch_end(self, batch, logs={}):
#         scores = self.model_.evaluate(
#             self.validation_images,
#             self.validation_labels,
#             verbose=0
#         )
#         print('\n', scores, '\n')
#         self.loss.append(logs.get('loss'))
#         self.accuracy.append(logs.get('accuracy'))
#         self.val_loss.append(scores[0])
#         self.val_accuracy.append(scores[1])
# 
# 
# def train(model, train_images, train_labels, validation_images, validation_labels, batch_size, num_epochs, learning_rate, verbose):
#     opt = SGD(learning_rate)
#     model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])         # compile model
# 
#     history = History(model, validation_images, validation_labels)                              # train model
#     model.fit(
#         train_images,
#         train_labels,
#         batch_size=batch_size,
#         epochs=num_epochs,
#         callbacks=[history]
#     )
# 
#     if verbose:
#         plot_learning_curve(history.loss)
#         plot_accuracy_curve(history.accuracy, history.val_accuracy)
# 
# 
# def evaluate(model):
#     scores = model.evaluate(test_images, test_labels, verbose=1)            # evaluate model
#     print('Test loss:', scores[0])
#     print('Test accuracy:', scores[1])
# 
# 
# def predict(model, image_idx):
#     layer_names = ['conv1', 'conv2', 'conv3', 'conv4']          # names of layers from which we will take the output
#     num_features = 4                                            # number of feature maps to display per layer
# 
#     dataset = load_cifar()
#     #dataset['test_images'] = np.moveaxis(dataset['test_images'], 1, 3)
#     image = test_images[image_idx]
#     image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
#     pred = np.argmax(model.predict(image))
# 
#     plot_sample(dataset['test_images'][image_idx], classes[dataset['test_labels'][image_idx]], classes[pred])
# 
# #     # extracting the output and appending to outputs
# #     feature_maps = []
# #     for name in layer_names:
# #         tmp_model = Model(inputs=model.input, outputs=model.get_layer(name).output)
# #         feature_maps.append(tmp_model.predict(image))
# 
# #     fig, ax = plt.subplots(nrows=len(feature_maps), ncols=num_features, figsize=(20, 20))
# #     for i in range(len(feature_maps)):
# #         for z in range(num_features):
# #             ax[i][z].imshow(feature_maps[i][0, :, :, z])
# #             ax[i][z].set_title(layer_names[i])
# #             ax[i][z].set_xticks([])
# #             ax[i][z].set_yticks([])
# #     plt.savefig('feature_maps.png')
# 
# 
# def plot_weights(model):
#     for layer in model.layers:
#         if 'conv' in layer.name:
#             weights, _ = layer.get_weights()
#             plot_histogram(layer.name, np.reshape(weights, -1))
# 
# 
# if __name__ == '__main__':
# 
#     classes = [                                                             # CIFAR-10 classes
#         "airplane",
#         "automobile",
#         "bird",
#         "cat",
#         "deer",
#         "dog",
#         "frog",
#         "horse",
#         "ship",
#         "truck"
#     ]
# 
#     num_epochs = 1 #10                                                        # hyper parameters
#     learning_rate = 0.005
#     batch_size = 2000 #100
#     lam = 0.01
#     verbose = 1
# 
#     print('\n--- Loading mnist dataset ---')                                # load dataset
#     dataset = load_cifar()
# 
#     print('\n--- Processing the dataset ---')                               # pre process dataset
#     dataset = preprocess(dataset)
# 
#     train_images = np.moveaxis(dataset['train_images'], 1, 3)               # pre process data for keras
#     validation_images = np.moveaxis(dataset['validation_images'], 1, 3)
#     test_images = np.moveaxis(dataset['test_images'], 1, 3)
#     train_labels = to_categorical(dataset['train_labels'])
#     validation_labels = to_categorical(dataset['validation_labels'])
#     test_labels = to_categorical(dataset['test_labels'])
# 
#     if os.path.isfile('model.h5'):                                          # load model
#         print('\n--- Loading model ---')
#         model = load_model('model.h5')
#     else:                                                                   # build model
#         print('\n--- Building model ---')
#         model = Sequential()
#         model.add(Conv2D(32, 3, name='conv1', activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(lam), input_shape=(32, 32, 3)))
#         model.add(Conv2D(32, 3, name='conv2', activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(lam)))
#         model.add(MaxPooling2D(2, name='pool1'))
#         model.add(Conv2D(64, 3, name='conv3', activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(lam)))
#         model.add(Conv2D(64, 3, name='conv4', activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(lam)))
#         model.add(MaxPooling2D(2, name='pool2'))
#         model.add(Flatten())
#         model.add(Dense(256, name='fullyconnected', activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(lam)))
#         model.add(Dense(10, name='dense', activation='softmax'))
#         
#         train_images = np.moveaxis(train_images, -1, 1)
#         validation_images = np.moveaxis(validation_images, -1, 1)
#         test_images = np.moveaxis(test_images, -1, 1)
# 
# 
#     train(
#         model,
#         train_images,
#         train_labels,
#         validation_images,
#         validation_labels,
#         batch_size,
#         num_epochs,
#         learning_rate,
#         verbose
#     )
# 
#     print('\n--- Testing the model ---')                                    # test model
#     evaluate(model)
# 
#     print('\n--- Predicting image from test set ---')
#     image_idx = 40                                                          # index of image to predict
#     predict(model, image_idx)
# 
#     print('\n--- Plotting weight distributions ---')
#     plot_weights(model)
# 
#     print('\n--- Saving the model ---')                                     # save model
# #     model.save('model.h5')

# === AFTER (edited) ===
def predict(model, image_idx):
    layer_names = ['conv1', 'conv2', 'conv3', 'conv4']
    num_features = 4

    dataset = load_cifar()

    image = test_images[image_idx]
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    pred = np.argmax(model.predict(image))

    plot_sample(dataset['test_images'][image_idx], classes[dataset['test_labels'][image_idx][0]], classes[pred])