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

#data descaler in real space
def data_descaler_real(data, max_data, min_data):
    data = (max_data - min_data)*data + min_data
    return 1e1**data


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

    losshistory, train_state = model.train(iterations=60000)
    model.compile("L-BFGS")
    losshistory, train_state = model.train()

    # Plot and save results
    #dde.utils.plot_loss_history(losshistory)
    #plt.savefig('loss_mesaemulator.png')
    model.save('model_mesa/model', verbose=1)



elif args.mode == "predict":

    eps = 0.1  # Tolerance for relative differences

    # Restore the model checkpoint
    model.restore('model_mesa/model-76336.ckpt', verbose=1)

    # Make predictions
    y_pred = model.predict(X_train)

    # Use a scientific colormap (e.g., viridis for better visibility in publications)
    colormap = plt.cm.viridis
    colors = [colormap(i) for i in np.linspace(0, 1, 53)]

    # Plot for mass
    fig, axs = plt.subplots(2, 1, gridspec_kw={'height_ratios': [4, 1]}, figsize=(10, 8))

    for i in range(N_data):
        # Main plot: predictions vs true values for mass
        axs[0].plot(
            timesteps,
            data_descaler(y_pred[i, :, 0], max_mass, min_mass),
            lw=1.5,
            color=colors[i],
            label=f"Sample {i+1}" if i < 10 else "_nolegend_"  # Limit legend clutter
        )
        axs[0].plot(
            timesteps,
            data_descaler(y_train[i, :, 0], max_mass, min_mass),
            lw=1.5,
            color=colors[i],
            linestyle='--'
        )

        # Subplot: relative differences
        relative_diff = (
            data_descaler_real(y_pred[i, :, 0], max_mass, min_mass) -
            data_descaler_real(y_train[i, :, 0], max_mass, min_mass)
        ) / (data_descaler_real(y_train[i, :, 0], max_mass, min_mass) + eps)

        axs[1].plot(timesteps, relative_diff, lw=1.5, color=colors[i])

    # Customize the main mass plot
    axs[0].set_ylabel(r'Log($M_{\odot}$)', fontsize=14)  # Use LaTeX for labels
    axs[0].set_xlabel('Evolution stage', fontsize=14)
    axs[0].set_title('Mass Prediction vs True', fontsize=16)
    axs[0].grid(visible=True, linestyle='--', linewidth=0.5)
    #axs[0].set_xlim(0.7) # optional

    handles = [
    plt.Line2D([0], [0], color='black', lw=1.5, label='Predicted (solid)'),
    plt.Line2D([0], [0], color='black', lw=1.5, linestyle='--', label='True (dashed)')]

    axs[0].legend(handles=handles, loc='lower left', fontsize=12)

    # Customize the relative difference subplot
    axs[1].set_ylabel('Relative Difference', fontsize=12)
    axs[1].set_xlabel('Evolution stage', fontsize=14)
    axs[1].axhline(0, color='black', lw=1, linestyle='--')
    axs[1].grid(visible=True, linestyle='--', linewidth=0.5)
    #axs[1].set_xlim(0.7) # optional
    # Adjust layout and save figure
    plt.tight_layout()
    plt.savefig('./plots/pred_vs_true_mass_zoom.png', dpi=300)  # Save with high resolution
    plt.close()

    # Plot for radius
    fig, axs = plt.subplots(2, 1, gridspec_kw={'height_ratios': [4, 1]}, figsize=(10, 8))

    for i in range(N_data):
        # Main plot: predictions vs true values for radius
        axs[0].plot(
            timesteps,
            data_descaler(y_pred[i, :, 1], max_radius, min_radius),
            lw=1.5,
            color=colors[i],
            label=f"Sample {i+1}" if i < 10 else "_nolegend_"
        )
        axs[0].plot(
            timesteps,
            data_descaler(y_train[i, :, 1], max_radius, min_radius),
            lw=1.5,
            color=colors[i],
            linestyle='--'
        )

        # Subplot: relative differences in real space
        relative_diff = (
            data_descaler_real(y_pred[i, :, 1], max_radius, min_radius) -
            data_descaler_real(y_train[i, :, 1], max_radius, min_radius)
        ) / (data_descaler_real(y_train[i, :, 1], max_radius, min_radius) + eps)

        axs[1].plot(timesteps, relative_diff, lw=1.5, color=colors[i])
        
    # Customize the main radius plot
    axs[0].set_ylabel(r'Log($R_{\odot}$)', fontsize=14)
    axs[0].set_xlabel('Evolution stage', fontsize=14)
    axs[0].set_title('Radius Prediction vs True', fontsize=16)
    axs[0].grid(visible=True, linestyle='--', linewidth=0.5)

    handles = [
    plt.Line2D([0], [0], color='black', lw=1.5, label='Predicted (solid)'),
    plt.Line2D([0], [0], color='black', lw=1.5, linestyle='--', label='True (dashed)')]

    axs[0].legend(handles=handles, loc='lower left', fontsize=12)
    #axs[0].set_xlim(0.7) #optional
    # Customize the relative difference subplot
    axs[1].set_ylabel('Relative Difference', fontsize=12)
    axs[1].set_xlabel('Evolution stage', fontsize=14)
    axs[1].axhline(0, color='black', lw=1, linestyle='--')
    axs[1].grid(visible=True, linestyle='--', linewidth=0.5)
    #axs[1].set_xlim(0.7) #optional
    # Adjust layout and save figure
    plt.tight_layout()
    plt.savefig('./plots/pred_vs_true_radius_zoom.png', dpi=300)
    plt.close()


    exit()
    # Heatmap for average relative differences, optional
    IC = np.loadtxt(data_dir + '/parameter_space.dat')[:, 1:]
    avg_relative_diff = np.zeros((len(IC[:,0]), len(IC[:,1])))

    for p1_idx, param1 in enumerate(IC[:,0]):
        for p2_idx, param2 in enumerate(IC[:,1]):
            # Compute average relative difference for the given parameter combination
            avg_diff = 0
            count = 0
            for i in range(N_data):
                relative_diff_mass = (data_descaler(y_pred[i,:,0], max_mass, min_mass) - data_descaler(y_train[i,:,0], max_mass, min_mass)) \
                                     / (data_descaler(y_train[i,:,0], max_mass, min_mass) + eps)
                relative_diff_radius = (data_descaler(y_pred[i,:,1], max_radius, min_radius) - data_descaler(y_train[i,:,1], max_radius, min_radius)) \
                                       / (data_descaler(y_train[i,:,1], max_radius, min_radius) + eps)
                avg_diff += np.mean(np.abs(relative_diff_mass)) + np.mean(np.abs(relative_diff_radius))
                count += 1
            avg_relative_diff[p1_idx, p2_idx] = avg_diff / count

    plt.figure(figsize=(8, 6))
    plt.imshow(avg_relative_diff, aspect='auto', origin='lower', 
            extent=[IC[0,0], IC[-1,0], IC[0,1], IC[-1,0]], cmap='inferno')
    plt.colorbar(label='Avg. Rel. Diff.')
    plt.xlabel('q')
    plt.ylabel('ai')
    plt.title('Heatmap of Avg. Relative Differences')
    plt.tight_layout()
    plt.savefig('./plots/heatmap_avg_relative_diff.png')
    plt.close()
