# DeepXDE MESA Emulator

import deepxde as dde
import matplotlib.pyplot as plt
import numpy as np

# Data scaler function
def data_scaler(data):
    data = np.log10(data)
    data = (data - np.amin(data)) / (np.amax(data) - np.amin(data))
    return data

# Define data directory
data_dir = '/leonardo_scratch/large/userexternal/lbranca0/MESAemulator/unaligned_data'

# Load and preprocess data
time = np.loadtxt(data_dir + '/time_grid_interp_unaligned.dat')
mass = data_scaler(np.load(data_dir + '/mass_interp_unaligned.npy'))
radius = data_scaler(np.load(data_dir + '/rad_interp_unaligned.npy'))

# Initial conditions
IC = np.loadtxt(data_dir + '/parameter_space.dat')[:, 1:]  # First column is constant (original mass)
IC[:, 0] = data_scaler(IC[:, 0])
IC[:, 1] = data_scaler(IC[:, 1])

# Time steps
timesteps = np.expand_dims(np.linspace(0, 1, len(time)), axis=-1)

# Training and testing data
X_train = (IC[:, :], timesteps)
X_test = (IC[0:10, :], timesteps)

y_train = np.stack([mass, radius], axis=-1)
y_test = np.stack([mass[0:10, :], radius[0:10, :]], axis=-1)

print(np.shape(y_train), np.shape(IC), np.shape(timesteps))

# Define data for DeepXDE
data = dde.data.Triple(
    X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
)

# DeepONet configuration
m = 2
dim_x = 1
regularization = ['l2', 0.0008]
activation = f"LAAF-{5} tanh"

net = dde.nn.DeepONetCartesianProd(
    [m, 32, 32],
    [dim_x, 32, 16],
    activation=activation,  # Adaptive activation
    kernel_initializer="Glorot normal",
    regularization=regularization,  # Important for sparse data
    num_outputs=2,
    multi_output_strategy="split_branch"
)

# Compile and train model
model = dde.Model(data, net)
model.compile("adam", lr=0.001, metrics=["mean l2 relative error"])
losshistory, train_state = model.train(iterations=10000)

# Plot and save results
dde.utils.plot_loss_history(losshistory)
plt.savefig('mesaemulator.png')
model.save('model_mesa/model', verbose=1)

