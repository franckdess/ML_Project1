def compute_loss(y, tx, w):
    e = y - tx@w
    mse = (-1/(2*y.size))*(e@e)
    return mse