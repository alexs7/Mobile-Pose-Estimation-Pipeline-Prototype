import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from tensorflow import estimator
from tensorflow_core.python.keras import Sequential
from tensorflow_core.python.keras.layers import Dense
from tensorflow_core.python.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import tensorflow_core.python.keras.backend as K

def soft_acc(y_true, y_pred):
    return K.mean(K.equal(K.round(y_true), K.round(y_pred)))

idxs =  [*range(1,210)]
training_data = np.empty([0, 129])

for idx in idxs:
    data = np.load("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/all_data_and_models/coop_local/ML_training_data/training_data_"+str(idx)+".npy")
    training_data = np.r_[training_data, data]

X = training_data[:, 0:128]
y = training_data[:, 128]

print("Data Preprocessing..")
sc = StandardScaler()
X = sc.fit_transform(X)

min_max_scaler = MinMaxScaler()
y = min_max_scaler.fit_transform(y.reshape(-1, 1))

# create model
print("Creating model")
model = Sequential()
model.add(Dense(128, input_dim=128, kernel_initializer='normal', activation='relu'))
model.add(Dense(32, input_dim=128, kernel_initializer='normal', activation='relu'))
model.add(Dense(1, kernel_initializer='normal'))

# Compile model
model.compile(optimizer='rmsprop', loss='mse', metrics=[soft_acc])

# train
print("Training..")
model.fit(X, y, epochs=15, batch_size=16)

print("Evaluate Model..")
test_data = np.load("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/all_data_and_models/coop_local/ML_training_data/training_data_last.npy")
X = test_data[:, 0:128]
y = test_data[:, 128]

X = sc.fit_transform(X)
y = min_max_scaler.fit_transform(y.reshape(-1, 1))

model.evaluate(X,y)

breakpoint()

# evaluation
# estimator = KerasRegressor(build_fn=model, epochs=100, batch_size=20, verbose=0)
# kfold = KFold(n_splits=10)
# results = cross_val_score(estimator, X, Y, cv=kfold)
# print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))