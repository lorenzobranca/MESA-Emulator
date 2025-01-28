# DeepXDE MESA Emulator for binaries evolution

This project provides a neural network-based emulator for the MESA (Modules for Experiments in Stellar Astrophysics) code using DeepXDE. The emulator predicts stellar parameters (mass and radius) across evolutionary stages and analyzes their relative differences from ground truth values.

## Features
- Neural Operator implementation using the DeepXDE library.
- Data preprocessing with scaling and descaling functions.
- Training and prediction modes for model evaluation.
- Visualization of predictions with respect to true values using Matplotlib.
- Heatmap generation for average relative differences.

## Prerequisites
- Python 3.8+
- Required Python libraries:
  - `deepxde`
  - `matplotlib`
  - `numpy`

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/lorenzobranca/MESA-Emulator.git
   cd MESA-Emulator
   ```
2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Modes
The emulator has two modes of operation:
- `train`: Train the DeepXDE model.
- `predict`: Use the trained model to make predictions and generate plots.

### Data Requirements
Prepare your input data in the following format:
- **Time grid:** `time_grid_interp_unaligned.dat`
- **Mass and radius interpolations:** `mass_interp_unaligned.npy`, `rad_interp_unaligned.npy`
- **Initial conditions:** `parameter_space.dat`

Save the data in a directory and update the `data_dir` variable in the code to point to it.

### Running the Emulator

#### Training the Model
Run the following command to train the model:
```bash
python emulator.py train
```
The model will be saved in the `model_mesa/` directory after training.

#### Making Predictions
Run the following command to predict and visualize results:
```bash
python emulator.py predict
```
Predictions, plots, and relative difference analyses will be saved in the `plots/` directory.

## File Structure
- `emulator.py`: Main script for training and prediction.
- `plots/`: Directory for storing generated plots.
- `model_mesa/`: Directory for saving trained model checkpoints.
- `requirements.txt`: List of required Python libraries.

## Outputs
1. **Plots for Mass and Radius Predictions:**
   - Comparison of predicted and true values.
   - Relative differences over time.
   - Saved as `pred_vs_true_mass.png`, `pred_vs_true_radius.png`, etc.

2. **Heatmap for Average Relative Differences:**
   - Visualizes the average relative differences for parameter combinations.
   - Saved as `heatmap_avg_relative_diff.png`.

## Code Highlights
- **Data Preprocessing:**
  Functions for scaling and descaling data:
  ```python
  def data_scaler(data):
      data = np.log10(data)
      max_data, min_data = np.amax(data), np.amin(data)
      data = (data - min_data) / (max_data - min_data)
      return data, max_data, min_data

  def data_descaler(data, max_data, min_data):
      data = (max_data - min_data) * data + min_data
      return data
  ```

- **Model Definition:**
  Configured using DeepXDE:
  ```python
  net = dde.nn.DeepONetCartesianProd(
      [m, 16, 16, 16, 16],
      [dim_x, 32, 32, 32, 32],
      activation="LAAF-10 tanh",
      kernel_initializer="Glorot normal",
      regularization=['l2', 0.0001],
      num_outputs=2,
      multi_output_strategy="split_trunk"
  )
  ```

## References
- [DeepXDE Documentation](https://deepxde.readthedocs.io/)
- [MESA](http://mesa.sourceforge.net/)

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

---

Feel free to open an issue or submit a pull request if you encounter any problems or have suggestions for improvement!

