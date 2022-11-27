import math
import data
import random
import numpy as np
from tqdm.auto import tqdm
import plotly.express as px

#######################################
# Helper Function
def sigmoid(x):
    return (1/(1 + np.exp(-x)))

def normalize_data(dataset):
    return (dataset - dataset.min()) / (dataset.max() - dataset.min())

def standardize_data(dataset):
    return (dataset - dataset.mean()) / dataset.std()

def initialize_weights_vector(input_size, nr_classes=10, nr_hlayers=300):
    W = np.random.normal(0, 1, size=(input_size, nr_hlayers))
    V = np.random.normal(0, 1, size=(nr_hlayers, nr_classes))
    b, c = np.array([0.] * nr_hlayers), np.array([0.] * nr_classes)

    return W, V, b, c

def convert_onehotencoding(y, t_index):
    # https://stackoverflow.com/questions/29816117/numpy-set-one-specific-element-of-each-column-based-on-indexing-by-array
    t = np.zeros(y.shape)
    t[np.arange(y.shape[0]), t_index] = 1
    return t
#######################################

#######################################
# Network
def update_weights(W, V, b, c, gradients, alpha=0.01):
    dV, dW, db, dc = gradients
    W = W - alpha*dW
    b = b - alpha*db
    V = V - alpha*dV
    c = c - alpha*dc

    return W, V, b, c

def compute_loss_acc(y, t_index):
    # pick the y value that corresponds to y_c
    loss = np.mean(-np.log(y[np.arange(y.shape[0]), t_index]))
    preds = list(np.argmax(y, axis=1))
    accuracy = np.mean(preds == t_index)
    return loss, accuracy

def forward_pass_vectorized(X, W, V, b, c):
    # linear
    k = np.dot(X, W) + b

    # sigmoid activation
    h = sigmoid(k)

    # linear
    o = np.dot(h, V) + c

    # softmax activation
    exp_o = np.exp(o)
    y = exp_o/np.sum(exp_o, axis=1, keepdims=True)

    context = k, h, o, y
    return context

def backward_pass_vectorized(X, t_index, V, context):
    k, h, o, y = context

    # compute d(Loss)/do[i]
    # https://davidbieber.com/snippets/2020-12-12-derivative-of-softmax-and-the-softmax-cross-entropy-loss/
    t = convert_onehotencoding(y, t_index)
    do = (y-t)

    # compute d(Loss)/dV[j][i] and d(Loss)/dh[i]
    dV = (h.T) @ do
    dh = do @ V.T
    dc = np.sum(do, axis=0)

    # compute d(Loss)/dk[i]
    dk = np.multiply(np.multiply(dh, h), (1-h))

    # compute d(Loss)/dW[i][j] and d(Loss)/db[j]
    dW = X.T @ dk
    db = np.sum(dk, axis=0)

    return dV, dW, db, dc

# train data
def train(x_train, t_index, x_val, t_val, num_mcls, nr_hlayers, epochs, alpha, batch_size=1, plot_batches=True):
    (W, V, b, c) = initialize_weights_vector(x_train.shape[1], nr_classes=num_mcls, nr_hlayers=nr_hlayers)

    # In this case mini batch becomes same as batch gradient descent
    if batch_size > x_train.shape[0]:
        batch_size = x_train.shape[0]

    num_batches = round(x_train.shape[0] / batch_size)
    batches, y_batches = np.array_split(x_train, num_batches), np.array_split(t_index, num_batches)

    epochs_loss_train, epochs_loss_val, epochs_batch_loss = [], [], []
    epochs_acc_train, epochs_acc_val, epochs_batch_acc = [], [], []

    for epoch in tqdm(range(epochs), desc='epochs'):
        for (batch_X, batch_t_index) in tqdm(zip(batches, y_batches), total=len(y_batches), desc='train', position=0):
            context = forward_pass_vectorized(batch_X, W, V, b, c)
            gradients = backward_pass_vectorized(batch_X, batch_t_index, V, context)

            (W, V, b, c) = update_weights(W, V, b, c, gradients, alpha=alpha)

            if plot_batches:
                # loss and accuracy for batch data
                context = forward_pass_vectorized(batch_X, W, V, b, c)
                _, _, _, pred = context
                loss, acc = compute_loss_acc(pred, batch_t_index)
                epochs_batch_loss.append(loss)
                epochs_batch_acc.append(acc)

        # train
        context = forward_pass_vectorized(x_train, W, V, b, c)
        _, _, _, pred_train = context
        loss_train, acc_train = compute_loss_acc(pred_train, t_index)
        epochs_loss_train.append(loss_train)
        epochs_acc_train.append(acc_train)

        # validation
        context = forward_pass_vectorized(x_val, W, V, b, c)
        _, _, _, pred_val = context
        loss_val, acc_val = compute_loss_acc(pred_val, t_val)
        epochs_loss_val.append(loss_val)
        epochs_acc_val.append(acc_val)

        print("Epoch {}: Train Loss: {} - Train Accuracy: {}".format(
            epoch, round(epochs_loss_train[epoch], 5), round(epochs_acc_train[epoch], 5)))
        print("Epoch {}: Validation Loss: {} - Validation Accuracy: {}".format(
            epoch, round(epochs_loss_val[epoch], 5), round(epochs_acc_val[epoch], 5)))

    return epochs_loss_train, epochs_acc_train, \
           epochs_loss_val, epochs_acc_val, \
           epochs_batch_loss, epochs_batch_acc
#######################################

# TEST
def test_vectorized_version():
    # Test vectorized version
    W, V = np.array([[1., 1., 1.], [-1, -1, -1]]), np.array([[1., 1.], [-1., -1.], [-1., -1.]])
    b, c = np.array([0, 0, 0]), np.array([0, 0])
    X, t_index = np.array([[1, -1]]), np.array([0])

    context = forward_pass_vectorized(X, W, V, b, c)
    gradients = backward_pass_vectorized(X, t_index, V, context)
    dV, dW, db, dc = gradients

    (W, V, b, c) = update_weights(W, V, b, c, gradients)

    # loss and accuracy for batch data
    context = forward_pass_vectorized(X, W, V, b, c)
    _, _, _, pred = context
    loss, acc = compute_loss_acc(pred, t_index)

    print('Loss: ', loss)
    print('Accuracy: ', acc)

    print("Gradient of W: ", dW)
    print("Gradient of b: ", db)
    print("Gradient of V: ", dV)
    print("Gradient of c: ", dc)

    print('W, V, b, c', W, V, b, c)

###
# Inputs: nr_hlayers, epochs, batch_size, alpha (which is the learning rate)
# Outputs: No outputs, print the final loss and acc for both training and test set- also plots the results
###
def run(dataset, alpha, batch_size, nr_hlayers=300, epochs=5):
    (x_train, y_train), (x_val, y_val), num_mcls = dataset
    x_train, x_val = normalize_data(x_train), normalize_data(x_val)  # normalize data

    epochs_loss_train, epochs_acc_train, \
    epochs_loss_val, epochs_acc_val, \
    epochs_batch_loss, epochs_batch_acc = \
        train(x_train, y_train, x_val, y_val, num_mcls, nr_hlayers, epochs, alpha, batch_size)

    print("Final Train Loss: {} - Final Train Accuracy: {}".format(
        round(epochs_loss_train[-1], 5), round(epochs_acc_train[-1], 5)))
    print("Final Validation Loss: {} - Final Validation Accuracy: {}".format(
        round(epochs_loss_val[-1], 5), round(epochs_acc_val[-1], 5)))

    return (epochs_loss_train, epochs_acc_train, \
            epochs_loss_val, epochs_acc_val, \
            epochs_batch_loss, epochs_batch_acc)

# main function to run different experiments for Mnist dataset
if __name__ == "__main__":
    # dataset = data.load_synth()
    dataset = data.load_mnist()

    # 0.
    # Experiment zero: Test different batch sizes

    # 1.
    # Experiment one: Compare the training loss per epoch to the validation loss per epoch
    # alpha, batch_size = 0.01, 1000
    # run(dataset, alpha, batch_size)

    # 2.
    # Experiment two: Test
    # Train the neural network from a random initialization multiple times (at least 3) and
    # plot an average and a standard deviation

    # 3.
    # Experiment three: Run the SGD with different learning rates
    # learning_rates = [0.001, 0.003, 0.01, 0.03]
    # alpha, batch_size = 0.01, 1000
    # for alpha in learning_rates:
        # run(dataset, alpha, batch_size)

    # 4.
    # train on the final network on the full training data and evaluate to the canonical test set
    # switch final to True
    final_parameters = 300, 5, 0.01, 550
    dataset_predict = data.load_mnist(final=True)
    print('dataset_predict.shape', dataset_predict[0][0].shape, dataset_predict[1][0].shape)
    nr_hlayers, epochs, alpha, batch_size = final_parameters
    run(dataset_predict, alpha=alpha, batch_size=batch_size, epochs=epochs)

