import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn import preprocessing

# Read the CSV file
data = pd.read_csv('creditcard.csv')

# Show the contents
# print(data)

# train = pd.merge(train_transaction, train_identity, on='TransactionID', how='left')
# test = pd.merge(test_transaction, test_identity, on='TransactionID', how='left')
# print(data.describe())

features = ['Amount'] + ['V%d' % number for number in range(1, 29)]

target = 'Class'

X = data[features]
y = data[target]

# Create a minimum and maximum processor object
# min_max_scaler = preprocessing.MinMaxScaler()


# Create an object to transform the data to fit minmax processor for the amount column
# the transform expects a 2 dimensional array, so use two brackets
# then replace the amount column with this data
# data['Amount'] = min_max_scaler.fit_transform(data[['Amount']])

def normalize(X):
    """
    Make the distribution of the values of each variable similar by subtracting the mean and by dividing by the standard deviation.
    """
    return preprocessing.scale(X)


# Define the model
model = LogisticRegression(solver='liblinear')

# Define the splitter for splitting the data in a train set and a test set
splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=0)

for train_indices, test_indices in splitter.split(X, y):
    # Select the train and test data
    X_train, y_train = X.iloc[train_indices], y.iloc[train_indices]
    X_test, y_test = X.iloc[test_indices], y.iloc[test_indices]

    # Normalize the data
    X_train = normalize(X_train)
    X_test = normalize(X_test)
    print(X_test)

    # Fit and predict!
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # And finally: show the results
    print(classification_report(y_test, y_pred))
