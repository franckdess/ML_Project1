def compute_gradient(y, tx, w):
    e = y - tx@w
    gradient = (-1/y.size)*(tx.T@e)
    return gradient