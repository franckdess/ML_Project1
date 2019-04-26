import numpy as np
import matplotlib.pyplot as plt

""" VIZUALIZATION CROSS VALIDATION"""

def cross_validation_visualization(lambds, mse_tr, mse_te):
    """visualization the curves of mse_tr and mse_te."""
    plt.semilogx(lambds, mse_tr, marker=".", color='b', label='train error')
    plt.semilogx(lambds, mse_te, marker=".", color='r', label='test error')
    plt.xlabel("lambda")
    plt.ylabel("rmse")
    plt.title("cross validation")
    plt.legend(loc=2)
    plt.grid(True)
    plt.savefig("cross_validation")
    
def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)

def cross_validation(y, x, k_indices, k, lambda_, degree):
    """return the loss of ridge regression."""
    # ***************************************************
    # get k'th subgroup in test, others in train: TODO
    
    test_indices = k_indices[k]
    train_indices = np.concatenate((k_indices[:k], k_indices[k+1:])).flatten()
    
    x_train = x[train_indices]
    y_train = y[train_indices]
    x_test = x[test_indices]
    y_test = y[test_indices]
    
    # ***************************************************
    # form data with polynomial degree: TODO
    m_train = build_poly(x_train, degree)
    m_test = build_poly(x_test, degree)
    
    # ***************************************************
    # ridge regression: TODO
    w_train, loss_train = ridge_regression(y_train, m_train, lambda_)
    
    # ***************************************************
    # calculate the loss for train and test data: TODO
    loss_tr = compute_mse(y_train, m_train, w_train)
    loss_te = compute_mse(y_test, m_test, w_train)

    return loss_tr, loss_te

def cross_validation_log_reg(y, x, k_indices, k, max_iters, gamma, degree):
    """return the loss of ridge regression."""
    # ***************************************************
    # get k'th subgroup in test, others in train: TODO
    
    test_indices = k_indices[k]
    train_indices = np.concatenate((k_indices[:k], k_indices[k+1:])).flatten()
    
    x_train = x[train_indices]
    y_train = y[train_indices]
    x_test = x[test_indices]
    y_test = y[test_indices]
    
    # ***************************************************
    # form data with polynomial degree: TODO
    m_train = build_poly(x_train, degree)
    m_test = build_poly(x_test, degree)
    
    # ***************************************************
    # ridge regression: TODO
    w_train, loss_train = logistic_regression(y_train, m_train, np.zeros(m_train.shape[1]), max_iters, gamma)
    h = sigmoid(m_test@w_train)
    loss_test = compute_loss_log_reg(h, y_test)
    # ***************************************************
    # calculate the loss for train and test data: TODO
    loss_tr = loss_train
    loss_te = loss_test

    return loss_tr, loss_te

def cross_validation_demo(y, x, degree):
    seed = 1
    k_fold = 4
    lambdas = np.logspace(-12, 0, 30)
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    # define lists to store the loss of training data and test data
    rmse_tr = []
    rmse_te = []
    # ***************************************************
    for lambda_ in lambdas:
        err_train = []
        err_test = []
        for k in range (k_fold):
            loss_tr, loss_te = cross_validation(y, x, k_indices, k, lambda_, degree) 
            err_train.append(np.sqrt(2*loss_tr))
            err_test.append(np.sqrt(2*loss_te))
        rmse_tr.append(sum(err_train)/k_fold)
        rmse_te.append(sum(err_test)/k_fold)
    # ***************************************************    
    cross_validation_visualization(lambdas, rmse_tr, rmse_te)
    
    
def cross_validation_demo_log_reg(y, x, max_iters, degree):
    seed = 1
    k_fold = 4
    gammas = np.logspace(-12, 0, 30)
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    # define lists to store the loss of training data and test data
    rmse_tr = []
    rmse_te = []
    # ***************************************************
    for gamma in gammas:
        err_train = []
        err_test = []
        for k in range (k_fold):
            loss_tr, loss_te = cross_validation_log_reg(y, x, k_indices, k, max_iters, gamma, degree)
            err_train.append(loss_tr)
            err_test.append(loss_te)
        rmse_tr.append(sum(err_train)/k_fold)
        rmse_te.append(sum(err_test)/k_fold)
    # ***************************************************    
    cross_validation_visualization(gammas, rmse_tr, rmse_te)

""" GENERAL FUNCTION """

def split_data_tr_te(x, y, ids, ratio, seed=1):
    """
    split the dataset based on the split ratio. If ratio is 0.8 
    you will have 80% of your data set dedicated to training 
    and the rest dedicated to testing
    """
    # set seed
    np.random.seed(seed)
    # ***************************************************
    # INSERT YOUR CODE HERE
    # split the data based on the given ratio: TODO
    n_row = y.size
    indices = np.random.permutation(n_row)
    
    split_idx = int(np.floor(ratio*n_row))
    
    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]
    
    x_train = x[train_idx]
    x_test = x[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]
    ids_train = ids[train_idx]
    ids_test = ids[test_idx]
    
    return x_train, x_test, y_train, y_test, ids_train, ids_test
    # ***************************************************
    
def predict_na_columns(x, indices):
    for i in indices:
        x_i_train, y_i_train, x_i_test, indices_to_drop_i, indices_to_keep_i = split_data(x, i, indices)
        w_i, loss_i = least_squares(y_i_train, x_i_train)
        y_i_test = x_i_test @ w_i
        y_i_arr = np.column_stack((indices_to_drop_i, y_i_test))
        x = put_back_y(x, i, y_i_arr)
    return x, w_i

def set_predict_na_columns(x, w, indices):
    for i in indices:
        x_i_train, y_i_train, x_i_test, indices_to_drop_i, indices_to_keep_i = split_data(x, i, indices)
        y_i_test = x_i_test @ w
        y_i_arr = np.column_stack((indices_to_drop_i, y_i_test))
        x = put_back_y(x, i, y_i_arr)
    return x

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

def get_na_columns(array, threshold, value):
    na_indices = []
    for ind, row in enumerate(array.T):
        count_na = 0
        for j in range(len(row)):
            if row[j] == value:
                count_na += 1
        if (count_na/len(row)) > threshold:
            na_indices.append(ind)
    return na_indices

def get_indices(vect):
    indices_to_drop = []
    indices_to_keep = []
    for i, value in enumerate(vect):
        if value == int(-999):
            indices_to_drop.append(i)
        else:
            indices_to_keep.append(i)
    return indices_to_drop, indices_to_keep

def build_poly(x, degree):
    poly = x
    for deg in range(2, degree+1):
        poly = np.concatenate((poly, np.power(x, deg)), axis = 1)
    return poly

def put_back_y(array, ind_col, y_fin):
    for ind, pair in enumerate(y_fin):
        index, valeur = pair
        index = int(index)
        array[index][ind_col] = valeur 
    return array

def split_data(x, indice, indices):
    y = x.T[indice].T
    indices_to_drop_i, indices_to_keep_i = get_indices(y)
    y_i_train = np.take(y, indices_to_keep_i)
    x_train_tr = np.delete(x, indices, axis = 1)
    x_i_train = np.take(x_train_tr, indices_to_keep_i, axis = 0)
    x_i_test = np.take(x_train_tr, indices_to_drop_i, axis = 0)
    return x_i_train, y_i_train, x_i_test, indices_to_drop_i, indices_to_keep_i
    
def replace_by_mean(x, mean, column_name):
    if x == -999:
        return mean
    else:
        return x
    
def neg_to_zero(array):
    ret = np.zeros(len(array))
    for i, v in enumerate(array):
        if v == -1:
            ret[i] = 0
        else:
            ret[i] = v
    return ret

def zero_to_neg(array):
    ret = np.zeros(len(array))
    for i, v in enumerate(array):
        if v == 0:
            ret[i] = -1
        else:
            ret[i] = v
    return ret

def standardize(x_train, x_test):
    mean = np.mean(x_train)
    norm = np.linalg.norm(x_train)
    x_train_std = (x_train - mean)/norm
    x_test_std = (x_test - mean)/norm
    return x_train_std, x_test_std

def least_squares(y, tx):
    A = tx.T@tx
    b = tx.T@y
    w = np.linalg.solve(A, b)
    loss = compute_mse(y, tx, w)
    return w, loss
    
def get_idx_of_line(array, line):
    for idx, row in enumerate(array):
        if(np.array_equal(row, line)):
            return idx
           
def get_util_col(y, x, actual_x, rem_x, actual_loss, indices_util):
    losses = []
    for column in (rem_x.T):
        x_t = np.column_stack((actual_x, column))
        w, loss = ridge_regression(y, x_t, 0.03)
        losses.append(loss)
    min_loss = np.min(losses)
    if((min_loss < actual_loss) and (len(rem_x.T) != 0)):
        actual_loss = min_loss
        idx_x = get_idx_of_line(losses, min_loss)
        vec_x = rem_x.T[idx_x]
        rem_x = np.append(rem_x.T[:idx_x], rem_x.T[idx_x+1:], axis=0).T
        actual_x = np.column_stack((actual_x, vec_x))
        i = get_idx_of_line(x.T, vec_x.T)
        indices_util.append(i)
        get_util_col(y, x, actual_x, rem_x, actual_loss, indices_util)
    return actual_x, indices_util   

""" RIDGE REGRESSION FUNCTIONS """

def ridge_regression(y, tx, lamb):
    """implement ridge regression."""
    aI = lamb * np.identity(tx.shape[1])
    a = tx.T.dot(tx) + aI
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    loss = compute_mse(y, tx, w)
    return w, loss

""" LEAST SQUARES FUNCTIONS """

def compute_mse(y, tx, w):
    e = y - tx.dot(w)
    mse = e.dot(e) / (2 * len(e))
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
    gradient = (tx[i])*(sigmoid(np.dot(tx, w))[i]-y[i])
    return gradient

def gradient_descent_log_reg(y, tx, initial_w, max_iters, gamma):
    w = initial_w
    for n_iter in range(max_iters):
        gradient = compute_gradient_log_reg(y, tx, w)
        h = sigmoid(tx@w)
        loss = compute_loss_log_reg(h, y)
        w = w - gamma * gradient
        #print("Gradient Descent({bi}/{ti}): loss={l}".format(
        #    bi=n_iter, ti=max_iters - 1, l=loss))
    return w, loss

def stoch_gradient_descent_log_reg(y, tx, initial_w, max_iters, gamma):
    w = initial_w
    for n_iter in range(max_iters):
        gradient = compute_stoch_gradient_log_reg(y, tx, w)
        h = sigmoid(tx@w)
        loss = compute_loss_log_reg(h, y)
        w = w - gamma * gradient
        #print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
        #     bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    return w, loss

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    w, loss = stoch_gradient_descent_log_reg(y, tx, initial_w, max_iters, gamma)
    return w, loss

""" REGULARIZED LOGISTIC REGRESSION FUNCTIONS """

def compute_loss_reg_log_reg(h, y, w, lambda_):
    epsilon = 1e-5
    loss = (-y * np.log(h + epsilon) - (1 - y) * np.log(1 - h + epsilon)).sum() + 2*(w.T@w)
    return loss

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    loss, w = reg_gradient_descent_log_reg(y, tx, lambda_, initial_w, max_iters, gamma)
    return w, loss

def reg_gradient_descent_log_reg(y, tx, lambda_, initial_w, max_iters, gamma):
    w = initial_w
    for n_iter in range(max_iters):
        gradient = compute_gradient_log_reg(y, tx, w) + 2*lambda_*w
        h = sigmoid(tx@w)
        loss = compute_loss_reg_log_reg(h, y, w, lambda_)
        w = w - gamma * gradient
        #print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
        #    bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    return w, loss