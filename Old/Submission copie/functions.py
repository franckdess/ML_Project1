import numpy as np
import matplotlib.pyplot as plt
from implementations import *

""" GENERAL FUNCTION """

# Sigmoid function that bring back x in the interval [0, 1]
def sigmoid(x):
    s = 0.5 * (1 + np.tanh(0.5*x))
    return s

# Given a matrix 'x' and the degree, return the polynomial expansion of 
# the matrix.
def build_poly(x, degree):
    poly = x
    for deg in range(2, degree+1):
        poly = np.concatenate((poly, np.power(x, deg)), axis = 1)
    return poly

# Given an array 'array', return a new array where the value -1 have
# been replaced by zero.
def neg_to_zero(array):
    ret = np.zeros(len(array))
    for i, v in enumerate(array):
        if v == -1:
            ret[i] = 0
        else:
            ret[i] = v
    return ret

# Given an array 'array', return a new array where the value zero have
# been replaced by -1.
def zero_to_neg(array):
    ret = np.zeros(len(array))
    for i, v in enumerate(array):
        if v == 0:
            ret[i] = -1
        else:
            ret[i] = v
    return ret

# Given train features and test features, return the standardized version
# of those array.
def standardize(x_train, x_test):
    mean = np.mean(x_train)
    norm = np.linalg.norm(x_train)
    x_train_std = (x_train - mean)/norm
    x_test_std = (x_test - mean)/norm
    return x_train_std, x_test_std

# Return the indices of the features of the array 'array' where the ratio
# of values 'value' is greater than threshold.
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

# Given the array of features x, predict the undefined values of the
# columns 'indices' of x, using least squares.
# Return x with the predicted values instead of the undefined values,
# and the models wi's used to find the predictions.
def predict_na_columns(x, indices):
    ws = []
    for i in indices:
        x_i_train, y_i_train, x_i_test, indices_to_drop_i, indices_to_keep_i = split_data(x, i, indices)
        w_i, loss_i = least_squares(y_i_train, x_i_train)
        y_i_test = x_i_test @ w_i
        y_i_arr = np.column_stack((indices_to_drop_i, y_i_test))
        x = put_back_y(x, i, y_i_arr)
        ws.append(w_i)
    return x, ws

# Given the array of features x, and the model w, predict the undefined 
# values of the columns 'indices' of x.
# Return x with the predicted values instead of the undefined values.
def set_predict_na_columns(x, w, indices):
    for i in indices:
        x_i_train, y_i_train, x_i_test, indices_to_drop_i, indices_to_keep_i = split_data(x, i, indices)
        y_i_test = x_i_test @ w
        y_i_arr = np.column_stack((indices_to_drop_i, y_i_test))
        x = put_back_y(x, i, y_i_arr)
    return x

# Given a vector 'vect', return the indices to drop, corresponding to the
# indices where the value is -999 in the vector and the indices to keep, 
# corresponding to the indices of the well defined values in the vector.
def get_indices(vect):
    indices_to_drop = []
    indices_to_keep = []
    for i, value in enumerate(vect):
        if value == int(-999):
            indices_to_drop.append(i)
        else:
            indices_to_keep.append(i)
    return indices_to_drop, indices_to_keep

# Given a matrix 'array', the indice of a column and a vector 'y_fin',
# replace the values of the columns by those of the vector 'y_fin'.
def put_back_y(array, ind_col, y_fin):
    for ind, pair in enumerate(y_fin):
        index, valeur = pair
        index = int(index)
        array[index][ind_col] = valeur 
    return array

# Given a matrix 'x', separate the array into x_train, x_test,
# y_train, y_test, in order to predict the undefined values of the
# matrix. It also returns the well defined value indices and the
# undefined value indices.
def split_data(x, indice, indices):
    y = x.T[indice].T
    indices_to_drop_i, indices_to_keep_i = get_indices(y)
    y_i_train = np.take(y, indices_to_keep_i)
    x_train_tr = np.delete(x, indices, axis = 1)
    x_i_train = np.take(x_train_tr, indices_to_keep_i, axis = 0)
    x_i_test = np.take(x_train_tr, indices_to_drop_i, axis = 0)
    return x_i_train, y_i_train, x_i_test, indices_to_drop_i, indices_to_keep_i
        
""" CROSS VALIDATION VIZUALIZATION """

# Displays the rmse of train and test set depending on the lambda.
def cross_validation_visualization(lambds, mse_tr, mse_te):
    plt.semilogx(lambds, mse_tr, marker=".", color='b', label='train error')
    plt.semilogx(lambds, mse_te, marker=".", color='r', label='test error')
    plt.xlabel("lambda")
    plt.ylabel("rmse")
    plt.title("cross validation")
    plt.legend(loc=2)
    plt.grid(True)
    
# Build k-indices for k-fold.
def build_k_indices(y, k_fold, seed):
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)

# Return the loss of ridge regression.
def cross_validation(y, x, k_indices, k, lambda_, degree):
    # get k'th subgroup in test, others in train: TODO
    test_indices = k_indices[k]
    train_indices = np.concatenate((k_indices[:k], k_indices[k+1:])).flatten()
    
    x_train = x[train_indices]
    y_train = y[train_indices]
    x_test = x[test_indices]
    y_test = y[test_indices]
    
    # form data with polynomial degree: TODO
    m_train = build_poly(x_train, degree)
    m_test = build_poly(x_test, degree)
    
    # ridge regression: TODO
    w_train, loss_train = ridge_regression(y_train, m_train, lambda_)
    
    # calculate the loss for train and test data: TODO
    loss_tr = compute_mse(y_train, m_train, w_train)
    loss_te = compute_mse(y_test, m_test, w_train)

    return loss_tr, loss_te

# Cross validation demo.
def cross_validation_demo(y, x, degree):
    seed = 1
    k_fold = 4
    lambdas = np.logspace(-12, 0, 30)
    
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    
    # define lists to store the loss of training data and test data
    rmse_tr = []
    rmse_te = []
    for lambda_ in lambdas:
        err_train = []
        err_test = []
        for k in range (k_fold):
            loss_tr, loss_te = cross_validation(y, x, k_indices, k, lambda_, degree) 
            err_train.append(np.sqrt(2*loss_tr))
            err_test.append(np.sqrt(2*loss_te))
        rmse_tr.append(sum(err_train)/k_fold)
        rmse_te.append(sum(err_test)/k_fold)
    cross_validation_visualization(lambdas, rmse_tr, rmse_te)