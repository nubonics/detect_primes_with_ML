import pickle
import pandas as pd

from joblib import dump, load
from sklearn import svm
from sklearn.model_selection import train_test_split


# Model filename
model_filename = 'prime_predictor.joblib'

# Read the positives (prime numbers) from the text file
positives = pd.read_csv('positives.txt', names=['integer'])
# Set the column `isprime` to True, as these integers are KNOWN to be primes
positives['isprime'] = 1

# Read the negatives (non prime numbers) from the text file
negatives = pd.read_csv('negatives.txt', names=['integer'])
# Set the column `isprime` to False, as these integers are KNOWN to NOT be primes
negatives['isprime'] = 0

# Merge the two dataframes together
isprimes = pd.concat([positives, negatives])

# y = LABELS
# Use all rows to label the data and use ONLY the label column to label the data
y = isprimes.iloc[:, 1]

# X = features
# Use all rows and all columns EXCEPT the label column as features
X = isprimes.iloc[:, :1]

# Split the data into training data, and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, shuffle=True)

# Create the type of model that is going to be used
SVM = svm.SVR()

# Fit the data
SVM.fit(X_train, y_train)

# Test the model
SVM.predict(X_test)

# Show the accuracy of the model
round(SVM.score(X, y), 4)

# Save the model
s = pickle.dumps(SVM)
dump(SVM, model_filename)

# Load the model
# SVM = load(model_filename)
