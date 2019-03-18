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
  num_train = X.shape[0]
  num_classes = W.shape[1]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i in range(num_train):
    score = X[i].dot(W)
    score -= np.max(score) # to avoid big number problem
    exp_score = np.exp(score)
    total = np.sum(exp_score)
    normal_score = exp_score / total
    loss += -np.log(normal_score[y[i]])
    for j in range(num_classes):    
       dW[:,j] += (normal_score[j] -(j == y[i])) * X[i] # numpy array broadcast
  
  loss /= num_train
  dW /= num_train

  loss += 0.5 * reg * np.sum(W*W)
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
  num_train = X.shape[0]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  score = X.dot(W)
  score -= np.max(score, axis=1, keepdims = True)
  exp_score = np.exp(score)
  sum_score = np.sum(exp_score, axis=1, keepdims = True)
  pred_score = exp_score / sum_score
  correct_score = pred_score[range(num_train),y].reshape(num_train,-1)
  loss = -np.sum(np.log(correct_score)) / num_train + 0.5 * reg * np.sum(W*W) 
  pred_score[range(num_train),y] -= 1
  dW = X.T.dot(pred_score) / num_train + reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

