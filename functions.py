import numpy as np

""" GENERAL FUNCTION """

def sigmoid(x):
    #s = 1/(1+np.exp(-x))
    s = 0.5 * (1 + np.tanh(0.5*x))
    return s

def get_column_mean(wrong_value, df, column_name):
    s = 0
    count = 0
    for idx, row in df.iterrows():
        if row[column_name] != wrong_value:
            count += 1
            s += row[column_name]
    mean = s/count
    return mean
    
def replace_by_mean(x, mean, column_name):
    if x == -999:
        return mean
    else:
        return x
    
def neg_to_zero(array):
    for i, v in enumerate(array):
        if v == -1:
            array[i] = 0
    return array

def zero_to_neg(array):
    for i, v in enumerate(array):
        if v == 0:
            array[i] = -1
    return array



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

def compute_loss_log_reg(h, y):
    epsilon = 1e-5
    loss = (-y * np.log(h + epsilon) - (1 - y) * np.log(1 - h + epsilon)).sum()
    return loss

def compute_gradient_log_reg(y, tx, w):
    gradient = (tx.T)@(sigmoid(tx@w)-y)
    return gradient

def compute_stoch_gradient_log_reg(y, tx, w):
    i = np.random.randint(0, len(y)-1)
    gradient = (tx[i])*(sigmoid(tx@w)[i]-y[i])
    return gradient

def gradient_descent_log_reg(y, tx, initial_w, max_iters, gamma):
    w = initial_w
    for n_iter in range(max_iters):
        gradient = compute_gradient_log_reg(y, tx, w)
        h = sigmoid(tx@w)
        loss = compute_loss_log_reg(h, y)
        w = w - gamma * gradient
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
             bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    return loss, w

def stoch_gradient_descent_log_reg(y, tx, initial_w, max_iters, gamma):
    w = initial_w
    for n_iter in range(max_iters):
        gradient = compute_stoch_gradient_log_reg(y, tx, w)
        h = sigmoid(tx@w)
        loss = compute_loss_log_reg(h, y)
        w = w - gamma * gradient
        #print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
        #     bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    return loss, w