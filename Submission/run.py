%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
from proj1_helpers import *
from functions import *
from implementations import *
%load_ext autoreload
%autoreload 2



train_set = load_csv_data('Data/train.csv', sub_sample = True)
test_set = load_csv_data('Data/test.csv', sub_sample = False)



x_train = train_set[1]
y_train = train_set[0]
ids_train = train_set[2]

x_test = test_set[1]
y_test = test_set[0]
ids_test = test_set[2]



complete_test = np.column_stack((y_test, x_test))
complete_test = np.column_stack((complete_test, ids_test))

subset_test_0 = complete_test[complete_test[:,23] == 0]
subset_test_1 = complete_test[complete_test[:,23] == 1]
subset_test_23 = complete_test[2 <= complete_test[:,23]]

y_test_0 = subset_test_0[:,0]
y_test_1 = subset_test_1[:,0]
y_test_2 = subset_test_23[:,0]

x_test_0 = subset_test_0[:,1:-1]
x_test_1 = subset_test_1[:,1:-1]
x_test_2 = subset_test_23[:,1:-1]

id_test_0 = subset_test_0[:,-1]
id_test_1 = subset_test_1[:,-1]
id_test_2 = subset_test_23[:,-1]



complete_train = np.column_stack((y_train, x_train))
complete_train = np.column_stack((complete_train, ids_train))

subset_train_0 = complete_train[complete_train[:,23] == 0]
subset_train_1 = complete_train[complete_train[:,23] == 1]
subset_train_23 = complete_train[2 <= complete_train[:,23]]

y_train_0 = subset_train_0[:,0]
y_train_1 = subset_train_1[:,0]
y_train_2 = subset_train_23[:,0]

x_train_0 = subset_train_0[:,1:-1]
x_train_1 = subset_train_1[:,1:-1]
x_train_2 = subset_train_23[:,1:-1]

id_train_0 = subset_train_0[:,-1]
id_train_1 = subset_train_1[:,-1]
id_train_2 = subset_train_23[:,-1]



na_indices_0 = get_na_columns(x_train_0, 0.95, -999)
na_indices_1 = get_na_columns(x_train_1, 0.95, -999)
na_indices_2 = get_na_columns(x_train_2, 0.95, -999)

x_train_0_clean = np.delete(x_train_0, na_indices_0, axis = 1)
x_train_1_clean = np.delete(x_train_1, na_indices_1, axis = 1)
x_train_2_clean = np.delete(x_train_2, na_indices_2, axis = 1)

x_test_0_clean = np.delete(x_test_0, na_indices_0, axis = 1)
x_test_1_clean = np.delete(x_test_1, na_indices_1, axis = 1)
x_test_2_clean = np.delete(x_test_2, na_indices_2, axis = 1)

zero_indices_0 = get_na_columns(x_train_0_clean, 0.99, 0.0)
zero_indices_1 = get_na_columns(x_train_1_clean, 0.99, 0.0)
zero_indices_2 = get_na_columns(x_train_2_clean, 0.99, 0.0)

x_train_0_clean = np.delete(x_train_0_clean, zero_indices_0, axis = 1)
x_train_1_clean = np.delete(x_train_1_clean, zero_indices_1, axis = 1)
x_train_2_clean = np.delete(x_train_2_clean, zero_indices_2, axis = 1)

x_test_0_clean = np.delete(x_test_0_clean, zero_indices_0, axis = 1)
x_test_1_clean = np.delete(x_test_1_clean, zero_indices_1, axis = 1)
x_test_2_clean = np.delete(x_test_2_clean, zero_indices_2, axis = 1)



na_indices_0_rem = get_na_columns(x_train_0_clean, 0, -999)
na_indices_1_rem = get_na_columns(x_train_1_clean, 0, -999)
na_indices_2_rem = get_na_columns(x_train_2_clean, 0, -999)



x_train_0_clean_pred, w_train_0 = predict_na_columns(x_train_0_clean, na_indices_0_rem)
x_train_1_clean_pred, w_train_1 = predict_na_columns(x_train_1_clean, na_indices_1_rem)
x_train_2_clean_pred, w_train_2 = predict_na_columns(x_train_2_clean, na_indices_2_rem)



x_test_0_clean_pred = set_predict_na_columns(x_test_0_clean, w_train_0[0], na_indices_0_rem)
x_test_1_clean_pred = set_predict_na_columns(x_test_1_clean, w_train_1[0], na_indices_1_rem)
x_test_2_clean_pred = set_predict_na_columns(x_test_2_clean, w_train_2[0], na_indices_2_rem)



x_train_0_std, x_test_0_std = standardize(x_train_0_clean_pred, x_test_0_clean_pred)
x_train_0_std_int = np.column_stack((np.ones(x_train_0_std.shape[0]), x_train_0_std))
x_test_0_std_int = np.column_stack((np.ones(x_test_0_std.shape[0]), x_test_0_std))

x_train_1_std, x_test_1_std = standardize(x_train_1_clean_pred, x_test_1_clean_pred)
x_train_1_std_int = np.column_stack((np.ones(x_train_1_std.shape[0]), x_train_1_std))
x_test_1_std_int = np.column_stack((np.ones(x_test_1_std.shape[0]), x_test_1_std))

x_train_2_std, x_test_2_std = standardize(x_train_2_clean_pred, x_test_2_clean_pred)
x_train_2_std_int = np.column_stack((np.ones(x_train_2_std.shape[0]), x_train_2_std))
x_test_2_std_int = np.column_stack((np.ones(x_test_2_std.shape[0]), x_test_2_std))



x_train_0_2 = build_poly(x_train_0_std_int, degree = 3)
x_train_1_2 = build_poly(x_train_1_std_int, degree = 3)
x_train_2_2 = build_poly(x_train_2_std_int, degree = 3)



w_0, loss_0 = ridge_regression(y_train_0, x_train_0_2, 10**(-11))
w_1, loss_1 = ridge_regression(y_train_1, x_train_1_2, 10**(-11))
w_2, loss_2 = ridge_regression(y_train_2, x_train_2_2, 10**(-11))



x_test_0_2 = build_poly(x_test_0_std_int, degree = 3)
x_test_1_2 = build_poly(x_test_1_std_int, degree = 3)
x_test_2_2 = build_poly(x_test_2_std_int, degree = 3)



y_0 = zero_to_neg(np.around(sigmoid(x_test_0_2 @ w_0)))
y_1 = zero_to_neg(np.around(sigmoid(x_test_1_2 @ w_1)))
y_2 = zero_to_neg(np.around(sigmoid(x_test_2_2 @ w_2)))



s_0 = np.column_stack((id_test_0, y_0))
s_1 = np.column_stack((id_test_1, y_1))
s_2 = np.column_stack((id_test_2, y_2))
s = np.vstack((np.vstack((s_0, s_1)), s_2))



y_pred = s[s[:,0].argsort()].astype(int)



create_csv_submission(y_pred[:,0], y_pred[:, 1], 'final_prediction.csv')