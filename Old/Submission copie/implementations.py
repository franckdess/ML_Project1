import numpy as np
import matplotlib.pyplot as plt

def least_squares(y, tx):
    A = tx.T@tx
    b = tx.T@y
    w = np.linalg.solve(A, b)
    loss = compute_mse(y, tx, w)
    return w, loss

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    ws = [initial_w]
    w = initial_w
    for n_iter in range(max_iters):
        gradient = compute_gradient_least_square(y, tx, w)
        loss = compute_mse(y, tx, w)
        w = w - gamma*gradient
    return w, loss

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):    
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
            gradient = compute_stoch_gradient(minibatch_y, minibatch_tx, w)
            loss = compute_mse(minibatch_y, minibatch_tx, w)
            new_w = w - gamma*gradient
            w = new_w
            ws.append(w)
            losses.append(loss)
    return ws[len(ws)-1], losses[len(losses)-1]

def ridge_regression(y, tx, lamb):
    aI = lamb * np.identity(tx.shape[1])
    a = tx.T.dot(tx) + aI
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    loss = compute_mse(y, tx, w)
    return w, loss

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    w, loss = gradient_descent_log_reg(y, tx, initial_w, max_iters, gamma)
    return w, loss

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    w, loss = reg_gradient_descent_log_reg(y, tx, lambda_, initial_w, max_iters, gamma)
    return w, loss


""" LEAST SQUARES FUNCTIONS """

# Given the prediction, the features and the model, return
# the mean squares error.
def compute_mse(y, tx, w):
    e = y - tx.dot(w)
    mse = e.dot(e) / (2 * len(e))
    return mse

# Given the prediction, the features and the model, compute
# the error and return the gradient for gradient descent
# least squares function.
def compute_gradient_least_square(y, tx, w):
    e = y - tx@w
    gradient = -tx.T@e / len(e)
    return gradient, e

# Given the prediction, the features and the model, compute
# the error and return the gradient for the stochastic 
# gradient descent least squares function.
def compute_stoch_gradient_least_square(y, tx, w):
    e = y - tx@w
    gradient = -tx.T@e / len(e)
    return gradient

""" RIDGE REGRESSION FUNCTIONS """

# Given the prediction, the features and the parameter lambda,
# return the model and the loss of ridge regression function.
def ridge_regression(y, tx, lamb):
    """implement ridge regression."""
    aI = lamb * np.identity(tx.shape[1])
    a = tx.T.dot(tx) + aI
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    loss = compute_mse(y, tx, w)
    return w, loss

""" LOGISTIC REGRESSION FUNCTIONS """

# Return the loss of the logistic regression.
def compute_loss_log_reg(h, y):
    epsilon = 1e-5
    loss = (-y * np.log(h + epsilon) - (1 - y) * np.log(1 - h + epsilon)).sum()
    return loss

# Return the gradient for the gradient descent logistic regression.
def compute_gradient_log_reg(y, tx, w):
    gradient = (tx.T)@(sigmoid(tx@w)-y)
    return gradient

# Return the gradient for the stochastic gradient descent 
# logistic regression.
def compute_stoch_gradient_log_reg(y, tx, w):
    i = np.random.randint(0, len(y)-1)
    gradient = (tx[i])*(sigmoid(np.dot(tx, w))[i]-y[i])
    return gradient

# Given the prediction, the features, the initial w, the maximum number of
# iterations and the parameter gamma, return the model and the loss for
# the gradient descent logistic regression.
def gradient_descent_log_reg(y, tx, initial_w, max_iters, gamma):
    w = initial_w
    for n_iter in range(max_iters):
        gradient = compute_gradient_log_reg(y, tx, w)
        h = sigmoid(tx@w)
        loss = compute_loss_log_reg(h, y)
        w = w - gamma * gradient
    return w, loss

# Given the prediction, the features, the initial w, the maximum number of
# iterations and the parameter gamma, return the model and the loss for
# the stochastic gradient descent logistic regression function.
def stoch_gradient_descent_log_reg(y, tx, initial_w, max_iters, gamma):
    w = initial_w
    for n_iter in range(max_iters):
        gradient = compute_stoch_gradient_log_reg(y, tx, w)
        h = sigmoid(tx@w)
        loss = compute_loss_log_reg(h, y)
        w = w - gamma * gradient
    return w, loss

# Return the model and the loss for the logistic regression function.
def logistic_regression(y, tx, initial_w, max_iters, gamma):
    w, loss = stoch_gradient_descent_log_reg(y, tx, initial_w, max_iters, gamma)
    return w, loss

""" REGULARIZED LOGISTIC REGRESSION FUNCTIONS """

# Return the loss of the regularized logistic regression.
def compute_loss_reg_log_reg(h, y, w, lambda_):
    epsilon = 1e-5
    loss = (-y * np.log(h + epsilon) - (1 - y) * np.log(1 - h + epsilon)).sum() + 2*(w.T@w)
    return loss

# Return the model and the loss for the regularized logistic regression function.
def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    loss, w = reg_gradient_descent_log_reg(y, tx, lambda_, initial_w, max_iters, gamma)
    return w, loss

# Return the model and the loss for the regularized gradient descent logistic
# regression function.
def reg_gradient_descent_log_reg(y, tx, lambda_, initial_w, max_iters, gamma):
    w = initial_w
    for n_iter in range(max_iters):
        gradient = compute_gradient_log_reg(y, tx, w) + 2*lambda_*w
        h = sigmoid(tx@w)
        loss = compute_loss_reg_log_reg(h, y, w, lambda_)
        w = w - gamma * gradient
    return w, loss