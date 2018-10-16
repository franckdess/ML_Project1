def compute_gradient(y, tx, w):
    e = y - tx@w
    gradient = (-1/y.size)*(tx.T@e)
    return gradient

def compute_loss(y, tx, w):
    e = y - tx@w
    mse = (-1/(2*y.size))*(e@e)
    return mse

def compute_stoch_gradient(y, tx, w):
    e = y - tx@w
    gradient = (-1/y.size)*np.transpose(tx)@e
    return gradient

def sigmoid(x):
    return 1/(1+np.exp(-x))