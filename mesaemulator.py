# DeepXDE MESA Emulator

import argparse
import deepxde as dde
import matplotlib.pyplot as plt
import numpy as np

# Data scaler function
def data_scaler(data):
    data = np.log10(data)

    max_data, min_data = np.amax(data), np.amin(data)

    data = (data - min_data) / (max_data - min_data)
    return data, max_data, min_data

# Data descaler function but in log
def data_descaler(data, max_data, min_data):
    data = (max_data - min_data)*data + min_data
    return data

N_data = 53

parser = argparse.ArgumentParser(description="train or predict.")
parser.add_argument("mode", choices=["train", "predict"], help="Exec mode: 'train' or 'predict'.")
args = parser.parse_args()

# Define data directory
data_dir = '/home/lollo/binaries/mesaemulator/unaligned_data'

# Load and preprocess data
time = np.loadtxt(data_dir + '/time_grid_interp_unaligned.dat')
mass, max_mass, min_mass       = data_scaler(np.load(data_dir + '/mass_interp_unaligned.npy'))
radius, max_radius, min_radius = data_scaler(np.load(data_dir + '/rad_interp_unaligned.npy'))

print('max and min mass log:', max_mass, min_mass, 'max and min radius log:',max_radius, min_radius)

# Initial conditions
IC = np.loadtxt(data_dir + '/parameter_space.dat')[:, 1:]  # First column is constant (original mass)
IC[:, 0], max_IC1, min_IC1 = data_scaler(IC[:, 0])
IC[:, 1], max_IC2, min_IC2 = data_scaler(IC[:, 1])

# Time steps
timesteps = np.expand_dims(np.linspace(0, 1, len(time)), axis=-1)

# Training and testing data
X_train = (IC[:, :], timesteps)
X_test = (IC[0:10, :], timesteps)

y_train = np.stack([mass[:,:], radius[:,:]], axis=-1)
y_test = np.stack([mass[0:10, :], radius[0:10, :]], axis=-1)

print(np.shape(y_train), np.shape(IC), np.shape(timesteps))

# Define data for DeepXDE
data = dde.data.Triple(
    X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
)

# DeepONet configuration
m = 2
dim_x = 1
regularization = ['l2', 0.0001]
activation = f"LAAF-{10} tanh"

net = dde.nn.DeepONetCartesianProd(
    [m, 16, 16, 16, 16],
    [dim_x, 32, 32, 32,  32],
    activation=activation,  # Adaptive activation
    kernel_initializer="Glorot normal",
    regularization=regularization,  # Important for sparse data
    num_outputs=2,
    multi_output_strategy="split_trunk"
)

# Compile 
model = dde.Model(data, net)
model.compile("adam", lr=0.001, metrics=["mean l2 relative error"])



if args.mode == "train":

    losshistory, train_state = model.train(iterations=30000)
    model.compile("L-BFGS")
    losshistory, train_state = model.train()

    # Plot and save results
    #dde.utils.plot_loss_history(losshistory)
    #plt.savefig('loss_mesaemulator.png')
    model.save('model_mesa/model', verbose=1)



elif args.mode == "predict":

    eps = 0.1 # tolerance in log

    model.restore('model_mesa/model-46109.ckpt', verbose=1)

    y_pred = model.predict(X_train)

    colormap = plt.cm.jet
    colors = [colormap(i) for i in np.linspace(0, 1, 53)]

    # Plot for mass
    fig, axs = plt.subplots(2, 1, gridspec_kw={'height_ratios': [4, 1]}, figsize=(10, 8))

    for i in range(N_data):
        # Main plot: predictions vs true values for mass
        axs[0].plot(timesteps, data_descaler(y_pred[i,:,0], max_mass, min_mass), lw=2, color=colors[i])
        axs[0].plot(timesteps, data_descaler(y_train[i,:,0], max_mass, min_mass), lw=3, color=colors[i], ls=':')

        # Subplot: relative differences
        relative_diff = (data_descaler(y_pred[i,:,0], max_mass, min_mass) - data_descaler(y_train[i,:,0], max_mass, min_mass)) \
                        / (data_descaler(y_train[i,:,0], max_mass, min_mass) + eps) 
        axs[1].plot(timesteps, relative_diff, lw=2, color=colors[i])

    # Finalize mass plot
    axs[0].set_ylabel('Log(Msun)')
    axs[0].set_xlabel('Evolution stage')
    axs[0].set_title('Mass Prediction vs True')

    axs[1].set_ylabel('Rel. Diff.')
    axs[1].set_xlabel('Evolution stage')
    axs[1].axhline(0, color='black', lw=1, ls='--')

    plt.tight_layout()
    plt.savefig('./plots/pred_vs_true_mass.png')
    plt.close()

    # Plot for radius
    fig, axs = plt.subplots(2, 1, gridspec_kw={'height_ratios': [4, 1]}, figsize=(10, 8))

    for i in range(N_data):
        # Main plot: predictions vs true values for radius
        axs[0].plot(timesteps, data_descaler(y_pred[i,:,1], max_radius, min_radius), lw=2, color=colors[i])
        axs[0].plot(timesteps, data_descaler(y_train[i,:,1], max_radius, min_radius), lw=3, color=colors[i], ls=':')

        # Subplot: relative differences
        relative_diff = (data_descaler(y_pred[i,:,1], max_radius, min_radius) - data_descaler(y_train[i,:,1], max_radius, min_radius)) \
                        / (data_descaler(y_train[i,:,1], max_radius, min_radius) + eps)
        axs[1].plot(timesteps, relative_diff, lw=2, color=colors[i])

    # Finalize radius plot
    axs[0].set_ylabel('Log(Rsun)')
    axs[0].set_xlabel('Evolution stage')
    axs[0].set_title('Radius Prediction vs True')

    axs[1].set_ylabel('Rel. Diff.')
    axs[1].set_xlabel('Evolution stage')
    axs[1].axhline(0, color='black', lw=1, ls='--')

    plt.tight_layout()
    plt.savefig('./plots/pred_vs_true_radius.png')
    plt.close()


