import numpy as np
from random import shuffle

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

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_classes = W.shape[1]
  num_train = X.shape[0]
  for i in range(num_train):
    scores = X[i].dot(W)
    scores -= np.max(scores)
    probabilites_of_classes = np.exp(scores)/np.sum(np.exp(scores))
    for j in range(num_classes):
      if j == y[i]:
        dW[:,j] += -1 * X[i] *(1 - probabilites_of_classes[j])
      else:
        dW[:,j] += -1 * X[i] *(0 - probabilites_of_classes[j])
    loss += - np.log (probabilites_of_classes[y[i]])
  #avg. loss and grad
  loss /= float(num_train)
  dW /= float(num_train)
  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

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
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  N = X.shape[0]
  K = W.shape[1]
  scores = X.dot(W)
  scores -= np.max(scores,axis=1,keepdims=True)
  probabilites_of_classes = np.divide(np.exp(scores),np.sum(np.exp(scores), axis=1,keepdims=True))
  loss_vec = -1 * np.log (probabilites_of_classes[range(N),y])
  
  one_condition = np.zeros_like(probabilites_of_classes)
  one_condition[range(N),y] = 1
  dW = -1 * X.T.dot(one_condition - probabilites_of_classes)
  
  loss = np.sum(loss_vec) / float(N)
  dW /= float(N)

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

