import numpy as np
import sys
import random

"""This script implements a two-class logistic regression model.
"""


class logistic_regression(object):

    def __init__(self, learning_rate, max_iter):
        self.learning_rate = learning_rate
        self.max_iter = max_iter

    def fit_BGD(self, X, y):
        """Train perceptron model on data (X,y) with Batch Gradient Descent.

        Args:
            X: An array of shape [n_samples, n_features].
            y: An array of shape [n_samples,]. Only contains 1 or -1.

        Returns:
            self: Returns an instance of self.
        """
        n_samples, n_features = X.shape

        ### YOUR CODE HERE
        self.assign_weights(np.random.normal(1, 1, size=(X.shape[1])) * 0.3)
        for iteration in range(self.max_iter):
            grads = np.zeros(n_features)
            for i in range(n_samples):
                grads += self._gradient(X[i],y[i])

            self.W = self.W - ((self.learning_rate * grads) / n_samples)
        ### END YOUR CODE
        return self

    def fit_miniBGD(self, X, y, batch_size):
        """Train perceptron model on data (X,y) with mini-Batch Gradient Descent.

        Args:
            X: An array of shape [n_samples, n_features].
            y: An array of shape [n_samples,]. Only contains 1 or -1.
            batch_size: An integer.

        Returns:
            self: Returns an instance of self.
        """
        ### YOUR CODE HERE
        self.assign_weights(np.random.normal(1, 1, size=(X.shape[1]))*0.3)
        batches = []
        n_features = X.shape[1]
        data = np.column_stack((X,y))
        number_of_batches = data.shape[0] // batch_size
        i = 0
        for i in range(number_of_batches):
            mini_batch = data[i*batch_size:(i+1)*batch_size,:]
            X_mini_batch = mini_batch[:,:-1]
            Y_mini_batch = mini_batch[:,-1].reshape((-1,1))
            batches.append((X_mini_batch,Y_mini_batch))
        if data.shape[0] % batch_size != 0:
            mini_batch = data[i*batch_size:data.shape[0]]
            X_mini_batch = mini_batch[:,:-1]
            Y_mini_batch = mini_batch[:,-1].reshape((-1,1))
            batches.append((X_mini_batch,Y_mini_batch))

        for iteration in range(self.max_iter):
            for m_b in batches:
                grads = np.zeros(n_features)
                X_mini, y_mini = m_b
                for i in range(batch_size):
                    grads = grads + self._gradient(X_mini[i],y_mini[i])

                self.W = self.W - ((self.learning_rate * grads / batch_size))
        ### END YOUR CODE
        return self

    def fit_SGD(self, X, y):
        """Train perceptron model on data (X,y) with Stochastic Gradient Descent.

        Args:
            X: An array of shape [n_samples, n_features].
            y: An array of shape [n_samples,]. Only contains 1 or -1.

        Returns:
            self: Returns an instance of self.
        """
        ### YOUR CODE HERE
        self.assign_weights(np.random.normal(1, 1, size=(X.shape[1])) * 0.3)
        for iteration in range(self.max_iter):
            idx_list = [i for i in range(X.shape[0])]
            random.shuffle(idx_list)
            for idx in idx_list:
                sample_X = X[idx]
                corresponding_y = y[idx]
                self.W = self.W - (self.learning_rate * (self._gradient(sample_X,corresponding_y)))
        ### END YOUR CODE
        return self

    def _gradient(self, _x, _y):
        """Compute the gradient of cross-entropy with respect to self.W
        for one training sample (_x, _y). This function is used in fit_*.

        Args:
            _x: An array of shape [n_features,].
            _y: An integer. 1 or -1.

        Returns:
            _g: An array of shape [n_features,]. The gradient of
                cross-entropy with respect to self.W.
        """
        ### YOUR CODE HERE
        h = np.dot(self.W,_x)
        log_reg = (1 + np.exp(_y * h))
        _g = -(_y*_x) / log_reg
        return _g
        ### END YOUR CODE

    def get_params(self):
        """Get parameters for this perceptron model.

        Returns:
            W: An array of shape [n_features,].
        """
        if self.W is None:
            print("Run fit first!")
            sys.exit(-1)
        return self.W

    def predict_proba(self, X):
        """Predict class probabilities for samples in X.

        Args:
            X: An array of shape [n_samples, n_features].

        Returns:
            preds_proba: An array of shape [n_samples, 2].
                Only contains floats between [0,1].
        """
    ### YOUR CODE HERE
        list_of_probabilities = []
        for sample in X:
            prediction = 1 / (1 + np.exp(-1 * (np.dot(sample, self.W))))
            prob_true_class = prediction
            prob_false_class = (1 - prob_true_class)
            list_of_probabilities.append([prob_true_class,prob_false_class])
        return list_of_probabilities
    ### END YOUR CODE

    def predict(self, X):
        """Predict class labels for samples in X.

        Args:
            X: An array of shape [n_samples, n_features].

        Returns:
            preds: An array of shape [n_samples,]. Only contains 1 or -1.
        """

    ### YOUR CODE HERE
        list_of_predictions = []
        for sample in X:
            prediction = 1 / (1 + np.exp(-1 * (np.dot(sample, self.W))))
            pred = 1 if (prediction > 0.5) else -1
            list_of_predictions.append(pred)
        return list_of_predictions
    ### END YOUR CODE

    def score(self, X, y):
        """Returns the mean accuracy on the given test data and labels.

        Args:
            X: An array of shape [n_samples, n_features].
            y: An array of shape [n_samples,]. Only contains 1 or -1.

        Returns:
            score: An float. Mean accuracy of self.predict(X) wrt. y.
        """
    ### YOUR CODE HERE
        packet = zip(X,y)
        accuracy = 0
        predictions = self.predict(X)
        for i in range(len(X)):
            equal = 1 if predictions[i] == y[i] else 0
            accuracy += equal
        accuracy = accuracy / len(y) *100
        return accuracy
    ### END YOUR CODE

    def assign_weights(self, weights):
        self.W = weights
        return self
