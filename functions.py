import numpy as np

""" GENERAL FUNCTION """

def sigmoid(x):
    #s = 1/(1+np.exp(-x))
    s = 0.5 * (1 + np.tanh(0.5*x))
    return s

""" LEAST SQUARES FUNCTIONS """

def compute_mse(y, tx, w):
    e = y - tx@w
    mse = (1/(2*len(e)))*(e@e)
    return mse

def compute_gradient_least_square(y, tx, w):
    e = y - tx@w
    gradient = -tx.T@e / len(e)
    return gradient, e

def compute_stoch_gradient_least_square(y, tx, w):
    e = y - tx@w
    gradient = -tx.T@e / len(e)
    return gradient

""" LOGISTIC REGRESSION FUNCTIONS """

def compute_gradient_log_reg(y, tx, w):
    gradient = (tx.T)@(sigmoid(tx@w)-y)
    return gradient

def gradient_descent_log_reg(y, tx, initial_w, max_iters, gamma):
    # Define parameters to store w and loss
    epsilon = 0.00001
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        # ***************************************************
        loss = 0
        gradient = compute_gradient_log_reg(y, tx, w)
        for i in range (len(y)):
            yn = y[i]
            xn = tx[i]
            loss += yn*np.log(sigmoid(xn.T@w) + epsilon) + (1-yn)*np.log(1-sigmoid(xn.T@w)+epsilon)
        losses.append(-loss)
        # ***************************************************
        # ***************************************************
        new_w = w - gamma*gradient
        w = new_w
        # ***************************************************
        # store w and loss
        ws.append(w)
        losses.append(loss)
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
             bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return losses, ws