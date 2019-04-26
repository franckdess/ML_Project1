{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import matplotlib.pyplot as plt"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "def least_squares(y, tx):\n",
    "    A = tx.T@tx\n",
    "    b = tx.T@y\n",
    "    w = np.linalg.solve(A, b)\n",
    "    loss = compute_mse(y, tx, w)\n",
    "    return w, loss"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "def least_squares_GD(y, tx, initial_w, max_iters, gamma):\n",
    "    ws = [initial_w]\n",
    "    w = initial_w\n",
    "    for n_iter in range(max_iters):\n",
    "        gradient = compute_gradient_least_square(y, tx, w)\n",
    "        loss = compute_mse(y, tx, w)\n",
    "        w = w - gamma*gradient\n",
    "    return w, loss"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "def least_squares_SGD(y, tx, initial_w, max_iters, gamma):\n",
    "    ws = [initial_w]\n",
    "    losses = []\n",
    "    w = initial_w\n",
    "    for n_iter in range(max_iters):    \n",
    "        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):\n",
    "            gradient = compute_stoch_gradient(minibatch_y, minibatch_tx, w)\n",
    "            loss = compute_mse(minibatch_y, minibatch_tx, w)\n",
    "            new_w = w - gamma*gradient\n",
    "            w = new_w\n",
    "            ws.append(w)\n",
    "            losses.append(loss)\n",
    "    return ws[len(ws)-1], losses[len(losses)-1]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "def ridge_regression(y, tx, lamb):\n",
    "    aI = lamb * np.identity(tx.shape[1])\n",
    "    a = tx.T.dot(tx) + aI\n",
    "    b = tx.T.dot(y)\n",
    "    w = np.linalg.solve(a, b)\n",
    "    loss = compute_mse(y, tx, w)\n",
    "    return w, loss"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "def logistic_regression(y, tx, initial_w, max_iters, gamma):\n",
    "    w, loss = gradient_descent_log_reg(y, tx, initial_w, max_iters, gamma)\n",
    "    return w, loss"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):\n",
    "    w, loss = reg_gradient_descent_log_reg(y, tx, lambda_, initial_w, max_iters, gamma)\n",
    "    return w, loss"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.1"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
