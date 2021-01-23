import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from tensorflow import estimator
from tensorflow_core.python.keras import Sequential
from tensorflow_core.python.keras.layers import Dense
from tensorflow_core.python.keras.wrappers.scikit_learn import KerasRegressor

dataset = np.loadtxt("/home/alex/fullpipeline/colmap_data/alfa_mega/slice1/points3D_sorted_descending_heatmap_per_image.txt")

X = dataset[:,4:132] #sift vectors
Y = dataset[:,3] #score

# create model
print("Creating model")
model = Sequential()
model.add(Dense(128, input_dim=128, kernel_initializer='normal', activation='relu'))
model.add(Dense(1, kernel_initializer='normal'))

# Compile model
model.compile(loss='mean_squared_error', optimizer='adam')

# train
print("Training..")
model.fit(X, Y, epochs=50)

breakpoint()

# evaluation
# estimator = KerasRegressor(build_fn=model, epochs=100, batch_size=20, verbose=0)
# kfold = KFold(n_splits=10)
# results = cross_val_score(estimator, X, Y, cv=kfold)
# print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))