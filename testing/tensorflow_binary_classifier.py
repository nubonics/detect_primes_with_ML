# Binary Classification with integer primes
import numpy as np
from pandas import read_csv, concat
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Read the positives (prime numbers) from the text file
positives = read_csv('../positives.txt', names=['integer'])
# Set the column `isprime` to True, as these integers are KNOWN to be primes
positives['isprime'] = 1

# Read the negatives (non prime numbers) from the text file
negatives = read_csv('../negatives.txt', names=['integer'])
# Set the column `isprime` to False, as these integers are KNOWN to NOT be primes
negatives['isprime'] = 0

# Merge the two dataframes together
isprimes = concat([positives, negatives])

# y = LABELS
# Use all rows to label the data and use ONLY the label column to label the data
y = isprimes.iloc[:, 1]

# X = features
# Use all rows and all columns EXCEPT the label column as features
X = isprimes.iloc[:, :1], -1

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(X)
encoded_Y = encoder.transform(y)


# larger model
def create_larger():
    # create model
    model = Sequential()
    model.add(Dense(60, input_dim=1, activation='relu'))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


estimators = [('standardize', StandardScaler()),
              ('mlp', KerasClassifier(build_fn=create_larger, epochs=5, batch_size=32, verbose=0))]
pipeline = Pipeline(estimators)
kfold = StratifiedKFold(n_splits=10, shuffle=True)
results = cross_val_score(pipeline, X, encoded_Y, cv=kfold)
print("Larger: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))
