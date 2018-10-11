def compute_stoch_gradient(y, tx, w):
    e = y - tx@w
    gradient = (-1/y.size)*np.transpose(tx)@e
    return gradient