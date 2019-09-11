import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn import preprocessing

# great solution: https://github.com/RoyMachineLearning/IEEE-CIS-Fraud-Detection/blob/master/LightGBM%20with%20Complete%20EDA%20-%20V2.0.ipynb

# Read the CSV file
train_transaction = pd.read_csv('train_transaction.csv')
test_transaction = pd.read_csv('test_transaction.csv')
train_identity = pd.read_csv('train_identity.csv')
test_identity = pd.read_csv('test_identity.csv')

# Show the contents
# print(data)

train = pd.merge(train_transaction, train_identity, on='TransactionID', how='left')
test = pd.merge(test_transaction, test_identity, on='TransactionID', how='left')
print(train.head(5))

# remove columns that have no data
train.dropna(inplace=True, axis=1)
test.dropna(inplace=True, axis=1)
print(train.head(5))


def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose: print(
        f'Mem. usage decreased to {end_mem:5.2f}Mb {(100 * (start_mem - end_mem) / start_mem):.1f}% reduction')
    return df


# this is the ternary operator or conditional like JS
a = 10
b = a if a > 10 else 11
print(b)

reduce_mem_usage(train)
reduce_mem_usage(test)

features = list(train.columns)
print(features)
# for number in range(1, 29)]
#
# target = 'Class'
#
# X = data[features]
# y = data[target]
#
# # Create a minimum and maximum processor object
# min_max_scaler = preprocessing.MinMaxScaler()
#
#
# # Create an object to transform the data to fit minmax processor for the amount column
# # the transform expects a 2 dimensional array, so use two brackets
# # then replace the amount column with this data
# # data['Amount'] = min_max_scaler.fit_transform(data[['Amount']])
#
# def normalize(X):
#     """
#     Make the distribution of the values of each variable similar by subtracting the mean and by dividing by the standard deviation.
#     """
#     return preprocessing.scale(X)
#
#
# # Define the model
# model = LogisticRegression(solver='liblinear')
#
# # Define the splitter for splitting the data in a train set and a test set
# splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=0)
#
# for train_indices, test_indices in splitter.split(X, y):
#     # Select the train and test data
#     X_train, y_train = X.iloc[train_indices], y.iloc[train_indices]
#     X_test, y_test = X.iloc[test_indices], y.iloc[test_indices]
#
#     # Normalize the data
#     X_train = normalize(X_train)
#     X_test = normalize(X_test)
#     print(X_test)
#
#     # Fit and predict!
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
#
#     # And finally: show the results
#     print(classification_report(y_test, y_pred))
