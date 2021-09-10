import keras
import _pickle as pickle
import numpy as np
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import matplotlib.pyplot as plt


# function to load data
def load_data(file_name):
    file_path = "cifar-10-batches-py/" + file_name

    print('Loading ' + file_name)
    with open(file_path, mode='rb') as file:
        data = pickle.load(file, encoding='bytes')
    raw_images = data[b'data']
    cls = np.array(data[b'labels'])

    images = raw_images.reshape([-1, 3, 32, 32])
    # move the channel dimension to the last
    images = np.rollaxis(images, 1, 4)

    return images, cls


# function to load test data
def load_test_data():
    images, cls = load_data(file_name="test_batch")
    return images, keras.utils.to_categorical(cls, 10)


# function to load train data, load one data_batch
def load_train_data():
    images, cls = load_data(file_name="data_batch_1")
    return images, keras.utils.to_categorical(cls, 10)


# function that splits the loaded train and validation sets
def load_cifar():
    X_train, Y_train = load_train_data()
    X_test, Y_test = load_test_data()

    return X_train, Y_train, X_test, Y_test


# declare batch size, classes and epochs
batch_size = 128
num_classes = 10
epochs = 10

# split data into train and validation sets
(x_train, y_train), (x_val, y_val) = cifar10.load_data()
y_train = keras.utils.to_categorical(y_train, num_classes)
y_val = keras.utils.to_categorical(y_val, num_classes)

# normalize input data
x_train = x_train.astype('float32')
x_val = x_val.astype('float32')
x_train = x_train / 255.0
x_val = x_val / 255.0

# reshape the data to appropriate tensor format
x_train = x_train.reshape(50000, 32, 32, 3)
x_val = x_val.reshape(10000, 32, 32, 3)

# print info
print(x_train.shape[0], 'train samples.')
print(x_val.shape[0], 'validation samples.')
print('We have', x_train.shape[0], 'training paradigms of size: ', x_train.shape[1], '*', x_train.shape[2], '.')

# define the model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(32, 32, 3)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
print('CNN topology setup completed.')

# print info about the model
model.summary()

# fit model parameters, given a set of training data
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_val, y_val))
print('Model has been trained.')

# saving model
model_name = 'CIFAR10_CNN.h5'
model.save(model_name)
print('Model has been saved.')

# print 4 images using the actual class as output
actual_class = np.argmax(y_train, axis=1)

class_to_demonstrate = 0
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
while sum(actual_class == class_to_demonstrate) > 4:
    tmp_idxs_to_use = np.where(actual_class == class_to_demonstrate)

    # create new plot window
    plt.figure()

    # plot 4 images
    plt.subplot(221)
    plt.imshow(x_train[tmp_idxs_to_use[0][0]])
    plt.subplot(222)
    plt.imshow(x_train[tmp_idxs_to_use[0][1]])
    plt.subplot(223)
    plt.imshow(x_train[tmp_idxs_to_use[0][2]])
    plt.subplot(224)
    plt.imshow(x_train[tmp_idxs_to_use[0][3]])
    tmp_title = 'Images considered as ' + str(class_names[class_to_demonstrate])
    plt.suptitle(tmp_title)

    # show the plot
    plt.show()
    plt.pause(2)

    # update the class to demonstrate index
    class_to_demonstrate = class_to_demonstrate + 1
