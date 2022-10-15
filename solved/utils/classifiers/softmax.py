"""
Implementation of softmax classifer.
"""

import numpy as np


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops over N samples)

    NOTE:
    In this function, you are NOT supposed to use functions like:
    - np.dot
    - np.matmul (or operator @)
    - np.linalg.norm
    You can (not necessarily) use functions like:
    - np.sum
    - np.log
    - np.exp

    Inputs have dimension D, there are K classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: a numpy array of shape (D + 1, K) containing weights.
    - X: a numpy array of shape (N, D + 1) containing a minibatch of data.
    - y: a numpy array of shape (N,) containing training labels; y[i] = k means 
        that X[i] has label k, where 0 <= k < K.
    - reg: regularization strength. For regularization, we use L2 norm.

    Returns a tuple of:
    - loss: the mean value of loss functions over N examples in minibatch.
    - gradient: gradient wrt W, an array of same shape as W
    """

    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    ############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.    #
    # Store the loss in loss and the gradient in dW. If you are not careful    #
    # here, it is easy to run into numeric instability. Don't forget the       #
    # regularization!                                                          #
    # NOTE: PLEASE pay attention to data types!                                #
    ############################################################################
    ############################################################################
    #                     START OF YOUR CODE                                   #
    ############################################################################
   
    Q = W.shape[1] 
    N = X.shape[0] 
    
    for i in range(N):
        f = np.dot(X[i,:],W)
    
        f      -= np.max(f)
        loss_i = -np.log(np.exp(f[y[i]])/np.sum(np.exp(f)))
        loss   +=loss_i 
        
        for j in range (Q):
            
            if j == y[i]:
                dW[:,j]+= (np.exp(f[j])/sum(np.exp(f)) - 1)*X[i] 
            else: 
                dW[:,j]+= (np.exp(f[j])/sum(np.exp(f)))*X[i]

    loss = loss/N + reg*np.sum(W*W)
    dW   = dW/N + 2*reg*W 
    # raise NotImplementedError
    ############################################################################
    #                     END OF YOUR CODE                                     #
    ############################################################################

    return loss, dW


def softmax(x):
    """
    Softmax function, vectorized version

    Inputs
    - x: (float) a numpy array of shape (N, C), containing the data

    Return a numpy array
    - h: (float) a numpy array of shape (N, C), containing the softmax of x
    """

    h = np.zeros_like(x)

    ############################################################################
    # TODO:                                                                    #
    # Implement the softmax function.                                          #
    # NOTE:                                                                    #
    # Carefully deal with different input shapes.                              #
    ############################################################################
    ############################################################################
    #                     START OF YOUR CODE                                   #
    ############################################################################
    e_x = np.exp(x - np.max(x))
    h = e_x / e_x.sum()
    # raise NotImplementedError
    ############################################################################
    #                     END OF YOUR CODE                                     #
    ############################################################################

    return h


def onehot(x, K):
    """
    One-hot encoding function, vectorized version.

    Inputs
    - x: (uint8) a numpy array of shape (N,) containing labels; y[i] = k means 
        that X[i] has label k, where 0 <= k < K.
    - K: total number of classes

    Returns a numpy array
    - y: (float) the encoded labels of shape (N, K)
    """

    N = x.shape[0]
    y = np.zeros((N, K))


    ############################################################################
    # TODO:                                                                    #
    # Implement the one-hot encoding function.                                 #
    ############################################################################
    ############################################################################
    #                     START OF YOUR CODE                                   #
    ############################################################################
    y[np.arange(N),x] = 1
    # raise NotImplementedError
    ############################################################################
    #                     END OF YOUR CODE                                     #
    ############################################################################

    return y


def cross_entropy(p, q):
    """
    Cross entropy function, vectorized version.

    Inputs:
    - p: (float) a numpy array of shape (N, K), containing ground truth labels
    - q: (float) a numpy array of shape (N, K), containing predicted labels

    Returns:
    - h: (float) a numpy array of shape (N,), containing the cross entropy of 
        each data point
    """

    h = np.zeros(p.shape[0])

    ############################################################################
    # TODO:                                                                    #
    # Implement cross entropy function.                                        #
    ############################################################################
    ############################################################################
    #                     START OF YOUR CODE                                   #
    ############################################################################

    # raise NotImplementedError
    ############################################################################
    #                     END OF YOUR CODE                                     #
    ############################################################################

    return h


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    NOTE:
    In this function, you CAN (not necessarily) use functions like:
    - np.dot (unrecommanded)
    - np.matmul (or operator @)
    - np.linalg.norm
    You MUST use the functions you wrote above:
    - onehot
    - softmax
    - crossentropy

    Inputs and outputs are the same as softmax_loss_naive.
    """

    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    ############################################################################
    # TODO: 																   #
	# Compute the softmax loss and its gradient using no explicit loops.       #
    # Store the loss in loss and the gradient in dW. If you are not careful    #
    # here, it is easy to run into numeric instability. Don't forget the       #
    # regularization!                                                          #
    ############################################################################
    ############################################################################
    #                     START OF YOUR CODE                                   #
    ############################################################################
    K = W.shape[1] 
    N = X.shape[0] 

    f = np.dot(X,W) 
    f = f - np.max(f, axis = 1).reshape(-1, 1)# normalize for numeric stability
    p = np.exp(f) / np.sum(np.exp(f), axis = 1).reshape(-1, 1)
    
    
    loss = - np.sum(np.log(p[np.arange(N), y]))
    
    loss = loss/N + reg *np.sum(W*W)
    
    
    p[np.arange(N), y] += -1
    dW = np.dot(X.T,p)
    
    dW = dW/N   + 2*reg*W
    # raise NotImplementedError
    ############################################################################
    #                     END OF YOUR CODE                                     #
    ############################################################################

    return loss, dW
