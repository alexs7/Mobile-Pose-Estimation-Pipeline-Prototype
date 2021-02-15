import os

import numpy as np
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from tensorflow import estimator
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import tensorflow.keras.backend as K
import sys
import glob
import pandas as pd

def soft_acc(y_true, y_pred):
    return K.mean(K.equal(K.round(y_true), K.round(y_pred)))

base_path = sys.argv[1]
no_csv = len(glob.glob1(base_path,"*.csv"))

target_index = 129 # score index
csv_file_index = -1

print("Loading data files..")
all_csvs = glob.glob(os.path.join(base_path, "*.csv"))
df_from_each_csv = (pd.read_csv(f) for f in all_csvs)
all_data = pd.concat(df_from_each_csv)

breakpoint()
print("Splitting data into test/train..")
X_train, X_test, y_train, y_test = train_test_split(all_data, all_data.iloc[:, target_index], test_size=0.2, shuffle=False)

X = X_train.iloc[:, 1:129]
y = y_train.values

print("Data Preprocessing..")
sc = StandardScaler()
X = sc.fit_transform(X)

min_max_scaler = MinMaxScaler()
y = min_max_scaler.fit_transform(y.reshape(-1, 1))

# create model
print("Creating model")
model = Sequential()
model.add(Dense(128, input_dim=128, kernel_initializer='normal', activation='relu')) #I know all input values will be positive at this point (SIFT)
model.add(Dense(32, input_dim=128, kernel_initializer='normal', activation='relu'))
model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))

# Compile model
model.compile(optimizer='rmsprop', loss='mse', metrics=[soft_acc])

# train
print("Training..")
model.fit(X, y, epochs=15, batch_size=1600)

print("Evaluate Model..")
X = X_test.iloc[:, 1:129]
y = y_test.values

X = sc.fit_transform(X)
y = min_max_scaler.fit_transform(y.reshape(-1, 1))

model.evaluate(X,y)

breakpoint()

# evaluation
# estimator = KerasRegressor(build_fn=model, epochs=100, batch_size=20, verbose=0)
# kfold = KFold(n_splits=10)
# results = cross_val_score(estimator, X, Y, cv=kfold)
# print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))