from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    for i in range(num_train):
        scores = X[i].dot(W)

        # compute the probabilities in numerically stable way
        scores -= np.max(scores)
        p = np.exp(scores)
        p /= p.sum()  # normalize
        logp = np.log(p)

        loss -= logp[y[i]]  # negative log probability is the loss


    # normalized hinge loss plus regularization
    loss = loss / num_train + reg * np.sum(W * W)

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    def train(self, X, y, learning_rate=1e-3, reg=1e-5, num_iters=100, batch_size=200, verbose=False):
        num_train, dim = X.shape
        num_classes = np.max(y) + 1
    
        if self.W is None:
            self.W = 0.001 * np.random.randn(dim, num_classes)
    
        loss_history = []
        for it in range(num_iters):
            batch_indices = np.random.choice(num_train, batch_size)
            X_batch = X[batch_indices]
            y_batch = y[batch_indices]

            loss, grad = self.loss(X_batch, y_batch, reg)
            loss_history.append(loss)

            self.W -= learning_rate * grad
        
            if verbose and it % 100 == 0:
                print(f'iteration {it}/{num_iters}: loss {loss:.4f}')
    
        return loss_history

    def predict(self, X):
        y_pred = np.argmax(X.dot(self.W), axis=1)
        return y_pred


    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)


    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the softmax loss, storing the           #
    # result in loss.                                                           #
    #############################################################################
    def softmax_loss_naive(W, X, y, reg):
        num_train = X.shape[0]  
        num_classes = W.shape[1]
        loss = 0.0
        dW = np.zeros_like(W)

        for i in range(num_train):
            scores = X[i].dot(W)
            scores -= np.max(scores) 
            exp_scores = np.exp(scores)
            probs = exp_scores / np.sum(exp_scores)
        
            loss += -np.log(probs[y[i]])
        
            dscores = probs.copy()
            dscores[y[i]] -= 1
            dW += np.outer(X[i], dscores)
    
        loss = loss / num_train + 0.5 * reg * np.sum(W * W)
        dW = dW / num_train + reg * W
    
    return loss, dW


    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the softmax            #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    def softmax_loss_vectorized(W, X, y, reg):
        num_train = X.shape[0] 
        scores = X.dot(W)
    

        scores -= np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(scores)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        correct_log_probs = -np.log(probs[np.arange(num_train), y])
        data_loss = np.sum(correct_log_probs) / num_train
        reg_loss = 0.5 * reg * np.sum(W * W)
        loss = data_loss + reg_loss
    
        dscores = probs.copy()
        dscores[np.arange(num_train), y] -= 1
        dW = X.T.dot(dscores) / num_train
        dW += reg * W 
    
        return loss, dW


    return loss, dW
