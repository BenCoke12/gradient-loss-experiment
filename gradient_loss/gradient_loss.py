#library imports
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import sys

#dataset imports for train datasets
datasetTrain = pd.read_csv("../data/pm_gradient_loss_train.csv")
print(datasetTrain.head())
datasetTrain['x'] = datasetTrain['x'].astype(np.float32)
datasetTrain['y'] = datasetTrain['y'].astype(np.float32)
datasetTrain['t0'] = datasetTrain['t0'].astype(np.float32)
datasetTrain['t1'] = datasetTrain['t1'].astype(np.float32)
datasetTrain['Nx'] = datasetTrain['Nx'].astype(np.float32)
datasetTrain['Ny'] = datasetTrain['Ny'].astype(np.float32)

train_xyt0t1 = datasetTrain
train_Nxy = pd.concat([datasetTrain.pop(x) for x in ['Nx', 'Ny']], axis=1)
print(train_xyt0t1.head())
print(train_Nxy.head())

#normalise/process data?
#convert to float32
print("type info:\n", type(train_xyt0t1.loc[0, 'y']))

#dataset imports for test datasets
datasetTest = pd.read_csv("../data/pm_gradient_loss_test.csv")
print(datasetTest.head())
datasetTest['x'] = datasetTest['x'].astype(np.float32)
datasetTest['y'] = datasetTest['y'].astype(np.float32)
datasetTest['t0'] = datasetTest['t0'].astype(np.float32)
datasetTest['t1'] = datasetTest['t1'].astype(np.float32)
datasetTest['Nx'] = datasetTest['Nx'].astype(np.float32)
datasetTest['Ny'] = datasetTest['Ny'].astype(np.float32)

test_xyt0t1 = datasetTest
test_Nxy = pd.concat([datasetTest.pop(x) for x in ['Nx', 'Ny']], axis=1)
print(test_xyt0t1.head())
print(test_Nxy.head())

#model definition and training
model = keras.Sequential()
model.add(keras.layers.Dense(32, activation=tf.nn.relu))
model.add(keras.layers.Dense(32, activation=tf.nn.relu))
model.add(keras.layers.Dense(32, activation=tf.nn.relu))
model.add(keras.layers.Dense(2, activation=tf.keras.activations.linear))

#datasets
batch_size = 32
train_dataset = tf.data.Dataset.from_tensor_slices((train_xyt0t1, train_Nxy))
test_dataset = tf.data.Dataset.from_tensor_slices((test_xyt0t1, test_Nxy))

train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
test_dataset = test_dataset.batch(batch_size)

#custom training loop
epochs = 25
optimizer = keras.optimizers.Adam()
mse_loss_fn = keras.losses.mean_squared_error
acc_metric = keras.metrics.MeanSquaredError()


mse_weight = 0.5
x_grad_weight = 0.25
y_grad_weight = 0.25

#gradient based loss functions
def grad_loss_x(x_batch, y_pred, y_batch):

    x1_pred = tf.gather(y_pred, [0], axis=1)
    x1_batch = tf.gather(y_batch, [0], axis=1)
    x0_batch = tf.gather(x_batch, [0], axis=1)
    t1_batch = tf.gather(x_batch, [3], axis=1)
    t0_batch = tf.gather(x_batch, [2], axis=1)

    deltat = tf.math.subtract(t1_batch, t0_batch)

    #gradient from true next point
    Deltax_true = tf.math.subtract(x1_batch, x0_batch)
    grad_true = tf.math.divide(Deltax_true, deltat)

    #gradient from predicted next point
    Deltax_pred = tf.math.subtract(x1_pred, x0_batch)
    grad_pred = tf.math.divide(Deltax_pred, deltat)

    #compute error how? - mse?
    grad_error = tf.keras.losses.mean_squared_error(grad_true, grad_pred)
    return grad_error

def grad_loss_y(x_batch, y_pred, y_batch):

    y1_pred = tf.gather(y_pred, [1], axis=1)
    y1_batch = tf.gather(y_batch, [1], axis=1)
    y0_batch = tf.gather(x_batch, [1], axis=1)
    t1_batch = tf.gather(x_batch, [3], axis=1)
    t0_batch = tf.gather(x_batch, [2], axis=1)

    deltat = tf.math.subtract(t1_batch, t0_batch)

    #gradient from true next point
    Deltay_true = tf.math.subtract(y1_batch, y0_batch)
    grad_true = tf.math.divide(Deltay_true, deltat)

    #gradient from predicted next point
    Deltay_pred = tf.math.subtract(y1_pred, y0_batch)
    grad_pred = tf.math.divide(Deltay_pred, deltat)

    #compute error how? - mse?
    grad_error = tf.keras.losses.mean_squared_error(grad_true, grad_pred)
    return grad_error


#training loop
for epoch in range(epochs):

    for batch_idx, (x_batch, y_batch) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            y_pred = model(x_batch, training=True)
            mse_loss = mse_loss_fn(y_batch, y_pred)
            x_grad_loss = grad_loss_x(x_batch, y_pred, y_batch)
            y_grad_loss = grad_loss_y(x_batch, y_pred, y_batch)

            weighted_loss = mse_weight * mse_loss + x_grad_weight * x_grad_loss + y_grad_weight * y_grad_loss

        gradients = tape.gradient(weighted_loss, model.trainable_weights)
        optimizer.apply_gradients(zip(gradients, model.trainable_weights))
        acc_metric.update_state(y_batch, y_pred)

    train_acc = acc_metric.result()
    print(f"Mean squared error over epoch {epoch}: {train_acc}")
    acc_metric.reset_states()

#test loop
for batch_idx, (x_batch, y_batch) in enumerate(test_dataset):
    y_pred = model(x_batch, training = False)
    acc_metric.update_state(y_batch, y_pred)

test_acc = acc_metric.result()
print(f"Mean squared error over test set: {test_acc}")
acc_metric.reset_states()

model.save("saved_models/gradient_loss")
