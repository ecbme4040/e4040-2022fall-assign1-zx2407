"""
Implementations of logistic regression. 
"""

import numpy as np


def logistic_regression_loss_naive(w, X, y, reg):
    """
    Logistic regression loss function, naive implementation (with loops over N samples)

    NOTE:
    In this function, you are NOT supposed to use functions like:
    - np.dot
    - np.matmul (or operator @)
    - np.linalg.norm
    You can (not necessarily) use functions like:
    - np.sum
    - np.log
    - np.exp

    Use this linear classification method to find optimal decision boundary.

    Inputs have dimension D, there are K classes, and we operate on minibatches
    of N examples.

    Inputs:
    - w: (float) a numpy array of shape (D + 1,) containing weights.
    - X: (float) a numpy array of shape (N, D + 1) containing a minibatch of data.
    - y: (uint8) a numpy array of shape (N,) containing training labels; y[i] = k means 
        that X[i] has label k, where k can be either 0 or 1.
    - reg: (float) regularization strength. For regularization, we use L2 norm.

    Returns a tuple of:
    - loss: (float) the mean value of loss functions over N examples in minibatch.
    - gradient: (float) gradient wrt W, an array of same shape as W
    """

    # Set the loss to a random number
    loss = 0
    # Initialize the gradient to zero
    dw = np.zeros_like(w)

    ############################################################################
    # TODO:                                                                    #
    # Compute the softmax loss and its gradient using explicit loops.          #
    # Store the loss in loss and the gradient in dW. If you are not careful    #
    # here, it is easy to run into numeric instability. Don't forget the       #
    # regularization!                                                          #
    # NOTE: You may want to convert y to float for computations. For numpy     #
    # dtypes, see https://numpy.org/doc/stable/reference/arrays.dtypes.html    #
    ############################################################################

    Q = w.shape[0]
    N = X.shape[0]

    f = np.dot(X, w)
    h = sigmoid(f)

    y0 = np.zeros([N,2])
    y0[np.arange(N),y] = 1
    for j in range(Q):
        dw[:,j] = np.dot(X.T,h[:,j]-y0[:,j]) 
        for i in range(N):   
            loss += -(y0[i,j]*np.log(h[i,j]) + (1-y0[i,j])*np.log(1-h[i,j]))
        
         
        
        

    loss = loss/N + reg*np.sum(w*w)
    dw   = dw/N + 2*reg*w

    return loss, dw


def sigmoid(x):
    """
    Sigmoid function.

    Inputs:
    - x: (float) a numpy array of shape (N,)

    Returns:
    - h: (float) a numpy array of shape (N,), containing the element-wise sigmoid of x
    """

    h = np.zeros_like(x)

    ############################################################################
    # TODO:                                                                    #
    # Implement sigmoid function.                                              #         
    ############################################################################
    ############################################################################
    h = 1/(1 + np.exp(-x))  

    return h  


def logistic_regression_loss_vectorized(w, X, y, reg):
    """
    Logistic regression loss function, vectorized version.

    NOTE:
    In this function, you CAN (not necessarily) use functions like:
    - np.dot (unrecommanded)
    - np.matmul
    - np.linalg.norm
    You MUST use the functions you wrote above:
    - sigmoid

    Use this linear classification method to find optimal decision boundary.

    Inputs and outputs are the same as softmax_loss_naive.
    """

    # Set the loss to a random number
    loss = 0
    # Initialize the gradient to zero
    dw = np.zeros_like(w)

    ############################################################################
    # TODO:                                                                    #
    # Compute the logistic regression loss and its gradient using no           # 
    # explicit loops.                                                          #
    # Store the loss in loss and the gradient in dW. If you are not careful    #
    # here, it is easy to run into numeric instability. Don't forget the       #
    # regularization!                                                          #
    # NOTE: For multiplication bewteen vectors/matrices, np.matmul(A, B) is    #
    # recommanded (i.e. A @ B) over np.dot see                                 #
    # https://numpy.org/doc/stable/reference/generated/numpy.matmul.html       #
    # Again, pay attention to the data types!                                  #
    ############################################################################
    ############################################################################
    K = w.shape[1]
    N = X.shape[0]

    f = np.dot(X, w)
    h = sigmoid(f)

    y0 = np.zeros([N,2])
    y0[np.arange(N),y] = 1
    
    loss = -np.sum(y0*np.log(h) + (1-y0)*np.log(1-h))

    dw = np.dot(X.T,h-y0)

    loss = loss/N + reg*np.sum(w*w)
    dw   = dw/N + 2*reg*w 

    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dw