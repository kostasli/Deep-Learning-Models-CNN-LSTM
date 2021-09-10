import _pickle as pickle
import matplotlib.pyplot as plt
import keras
from keras.datasets import cifar10
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, accuracy_score, recall_score

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


# function to load the test_batch
def load_test_data():
    images, cls = load_data(file_name="test_batch")
    return images, keras.utils.to_categorical(cls, 10)


# load the trained model
model_name = 'CIFAR10_CNN.h5'
loaded_model = keras.models.load_model(model_name)

# load test data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# Normalize the input data
x_test = x_test.astype('float32')
x_test = x_test / 255.0

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

y_test_predictions = loaded_model.predict(x_test)
y_test_predictions = np.argmax(y_test_predictions, axis=1)
y_test = np.argmax(y_test, axis=1)

# print 9 images using the CNN predictions as output

class_to_demonstrate = 0
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
while sum(y_test_predictions == class_to_demonstrate) > 4:
    tmp_idxs_to_use = np.where(y_test_predictions == class_to_demonstrate)

    # create new plot window
    plt.figure()

    # plot 4 images
    plt.subplot(221)
    plt.imshow(x_test[tmp_idxs_to_use[0][0]])
    plt.subplot(222)
    plt.imshow(x_test[tmp_idxs_to_use[0][1]])
    plt.subplot(223)
    plt.imshow(x_test[tmp_idxs_to_use[0][2]])
    plt.subplot(224)
    plt.imshow(x_test[tmp_idxs_to_use[0][3]])
    tmp_title = 'Images considered as ' + str(class_names[class_to_demonstrate])
    plt.suptitle(tmp_title)

    # show the plot
    plt.show()
    plt.pause(2)

    # update the class to demonstrate index
    class_to_demonstrate = class_to_demonstrate + 1

# print the appropriate scores
print('F1 score test:', f1_score(y_test, y_test_predictions, average='macro'))
print('Precision score test:', precision_score(y_test, y_test_predictions, average='macro'))
print('Accuracy score test:', accuracy_score(y_test, y_test_predictions))
print('Recall score test:', recall_score(y_test, y_test_predictions, average='macro'))
print('Confusion matrix: \n', confusion_matrix(y_test, y_test_predictions))
