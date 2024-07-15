#library imports
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import sys

#load model
model = tf.keras.models.load_model("saved_models/gradient_loss")

#dataset imports for train datasets
datasetTrain = pd.read_csv("../data/pm_gradient_loss_train.csv")
print(datasetTrain.head())
datasetTrain['x'] = datasetTrain['x'].astype(np.float64)
datasetTrain['y'] = datasetTrain['y'].astype(np.float64)
datasetTrain['t0'] = datasetTrain['t0'].astype(np.float64)
datasetTrain['t1'] = datasetTrain['t1'].astype(np.float64)
datasetTrain['Nx'] = datasetTrain['Nx'].astype(np.float64)
datasetTrain['Ny'] = datasetTrain['Ny'].astype(np.float64)

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
datasetTest['x'] = datasetTest['x'].astype(np.float64)
datasetTest['y'] = datasetTest['y'].astype(np.float64)
datasetTest['t0'] = datasetTest['t0'].astype(np.float64)
datasetTest['t1'] = datasetTest['t1'].astype(np.float64)
datasetTest['Nx'] = datasetTest['Nx'].astype(np.float64)
datasetTest['Ny'] = datasetTest['Ny'].astype(np.float64)

test_xyt0t1 = datasetTest
test_Nxy = pd.concat([datasetTest.pop(x) for x in ['Nx', 'Ny']], axis=1)
print(test_xyt0t1.head())
print(test_Nxy.head())

#test model on point in test data:
sample = 40
print("Test data:\n" + str(test_xyt0t1.iloc[sample]))
print("True Nxy: ", test_Nxy.iloc[sample])
x = test_xyt0t1.iloc[sample][0]
y = test_xyt0t1.iloc[sample][1]
t0 = test_xyt0t1.iloc[sample][2]
t1 = test_xyt0t1.iloc[sample][3]

inputs = tf.expand_dims(tf.convert_to_tensor([x, y, t0, t1]), 0)

print("Predicted Nx: ", float(model(inputs)[0][0]))
print("Predicted Ny: ", float(model(inputs)[0][1]))

#root means squared error
predictions = model.predict(test_xyt0t1)
rmse = np.sqrt(mean_squared_error(test_Nxy, predictions))
print("Root Mean Squared Error (RMSE):", rmse)


#Residual simulation

#initial conditions
x_0 = 0
y_0 = 0
u_x = 20
u_y = 20
g = -10
t_0 = 0

#positions lists
x_pos = [x_0]
y_pos = [y_0]
times = [t_0]

#delta t
step = 0.001
counter = 0

def network(x, y, t0, t1):
    inputs = tf.expand_dims(tf.convert_to_tensor([x, y, t0, t1]), 0)
    result = model(inputs)
    Nx = result[0][0]
    Ny = result[0][1]
    return (Nx, Ny)

max_iter = 12000
while counter < max_iter:
    if (y_pos[counter] < 0): break
#while (y_pos[counter] >= 0):
    #current position
    x_current = x_pos[counter]
    y_current = y_pos[counter]
    t0 = times[counter]

    timestep = step * np.random.random()
    t1 = t0 + timestep

    #run model
    result = network(x_current, y_current, t0, t1)
    x_next = result[0]
    y_next = result[1]

    x_pos.append(x_next)
    y_pos.append(y_next)
    times.append(t1)

    #update current count
    counter += 1

    #percent complete
    if(counter%(max_iter/100) == 0):
        sys.stdout.write('\r')
        sys.stdout.write("Percent complete: " + str(int(counter/(max_iter/100))) + "%")
        sys.stdout.flush()

#plot the trajectory
def plot_trajectory(x_pos, y_pos):
    plt.plot(x_pos, y_pos)
    plt.show()

plot_trajectory(x_pos, y_pos)
