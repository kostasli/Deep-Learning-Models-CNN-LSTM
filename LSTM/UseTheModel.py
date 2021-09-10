from random import randint
from numpy import array
from math import ceil
from math import log10
from numpy import argmax
import keras


# function to generate lists of random pairs of integers their sum and sub
def random_pairs(n_examples, n_numbers, largest):
    X, y = list(), list()
    for i in range(n_examples):
        in_pattern = [randint(1, largest) for _ in range(n_numbers)]

        out_pattern = sum(in_pattern)
        X.append(in_pattern)
        y.append(out_pattern)

        out_pattern = in_pattern[0] - in_pattern[1]
        X.append(in_pattern)
        y.append(out_pattern)
    return X, y


# # function to convert data to strings and add the operator
# # note that for the same pair two operations are conducted(+ and -)
def to_string(X, y, n_numbers, largest):
    max_length = n_numbers * ceil(log10(largest + 1)) + n_numbers - 1
    Xstr = list()
    i = 0
    for pattern in X:
        if sum(pattern) == y[i]:
            strp = '+'.join([str(n) for n in pattern])
            strp = ''.join([' ' for _ in range(max_length - len(strp))]) + strp
            Xstr.append(strp)
        else:
            strp = '-'.join([str(n) for n in pattern])
            strp = ''.join([' ' for _ in range(max_length - len(strp))]) + strp
            Xstr.append(strp)
        i = i + 1
    max_length = ceil(log10(n_numbers * (largest + 1)))
    ystr = list()
    for pattern in y:
        strp = str(pattern)
        strp = ''.join([' ' for _ in range(max_length - len(strp))]) + strp
        ystr.append(strp)
    return Xstr, ystr


# function to encode the integers
def integer_encode(X, y, alphabet):
    char_to_int = dict((c, i) for i, c in enumerate(alphabet))
    Xenc = list()
    for pattern in X:
        integer_encoded = [char_to_int[char] for char in pattern]
        Xenc.append(integer_encoded)
    yenc = list()
    for pattern in y:
        integer_encoded = [char_to_int[char] for char in pattern]
        yenc.append(integer_encoded)
    return Xenc, yenc


# function for one hot encode
def one_hot_encode(X, y, max_int):
    Xenc = list()
    for seq in X:
        pattern = list()
        for index in seq:
            vector = [0 for _ in range(max_int)]
            vector[index] = 1
            pattern.append(vector)
        Xenc.append(pattern)
    yenc = list()
    for seq in y:
        pattern = list()
        for index in seq:
            vector = [0 for _ in range(max_int)]
            vector[index] = 1
            pattern.append(vector)
        yenc.append(pattern)
    return Xenc, yenc


# generate an encoded dataset
def generate_data(n_samples, n_numbers, largest, alphabet):
    # generate pairs
    X, y = random_pairs(n_samples, n_numbers, largest)
    # convert to strings
    X, y = to_string(X, y, n_numbers, largest)
    for i in range (len(X)):
        print("Sequences:", X[i])
    # integer encode
    X, y = integer_encode(X, y, alphabet)
    # one hot encode
    X, y = one_hot_encode(X, y, len(alphabet))
    # return as numpy arrays
    X, y = array(X), array(y)
    return X, y


# invert encoding
def invert(seq, alphabet):
    int_to_char = dict((i, c) for i, c in enumerate(alphabet))
    strings = list()
    for pattern in seq:
        string = int_to_char[argmax(pattern)]
        strings.append(string)
    return ''.join(strings)


# load the trained model and test it with the test data
model_name = 'LSTM_add_sub.h5'
loaded_model = keras.models.load_model(model_name)

n_samples = 1000
n_numbers = 2
largest = 10
alphabet = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '-', ' ']
n_batch = 10
n_epoch = 30

# evaluate on some new patterns
X, y = generate_data(n_samples, n_numbers, largest, alphabet)
result = loaded_model.predict(X, batch_size=n_batch, verbose=0)
# calculate error
expected = [invert(x, alphabet) for x in y]
predicted = [invert(x, alphabet) for x in result]
# show some examples
for i in range(100):
    print('Expected=%s, Predicted=%s' % (expected[i], predicted[i]))

# test the model for 5 user inputs
for test in range(5):
    output = str(0)
    print('Enter a sequence (a+b or a-b):')
    sequence = input()
    # add whitespace in order to have length of 5 char
    if len(sequence) == 3:
        sequence = '  ' + sequence
    elif len(sequence) == 4:
        sequence = ' ' + sequence
    sequence, output = integer_encode(sequence, output, alphabet)  # encode the sequence
    sequence, output = one_hot_encode(sequence, output, len(alphabet))  # one hot encode the sequence
    sequence = array(sequence)  # make it type of ndarray
    sequence = sequence.reshape(1, 5, 13)  # reshape it
    # get the result
    result = loaded_model.predict(sequence, batch_size=n_batch, verbose=0)
    result = [invert(x, alphabet) for x in result]  # decode the result
    print("The result is: ", result[0])
