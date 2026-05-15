# --- [CELL 0]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 1}
# === BEFORE (original) ===
# import os
# import numpy as np
# import torch
# 
# # Function to load loss weights from .npy files in a directory
# def load_loss_weights_from_directory(directory_path):
#     weight_files = [filename for filename in os.listdir(directory_path) if filename.endswith(".npy")]
#     weights = [np.load(os.path.join(directory_path, filename)) for filename in weight_files]
#     return np.concatenate(weights)
# 
# # Function to save the updated weights to a directory
# def save_weights_to_directory(directory_path, weights):
#     os.makedirs(directory_path, exist_ok=True)  # Create the directory if it doesn't exist
#     np.save(os.path.join(directory_path, "updated_regression_weights.npy"), weights)
# 
# # Directory path for initial regression loss weights
# regression_weights_directory = 'data/adjusted_survival_2019'
# 
# # Load initial regression loss weights from the directory
# regression_weight = load_loss_weights_from_directory(regression_weights_directory)
# 
# # Training loop for regression task
# num_epochs_update_regression = 5  # Number of epochs to update the regression weight
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 
# # Create a new directory to save the updated regression weights
# output_directory = 'data/updated_regression_weights'
# os.makedirs(output_directory, exist_ok=True)
# 
# for epoch in range(num_epochs_update_regression):
#     # Calculate mean and standard deviation for the regression loss (replace this with your actual loss calculation)
#     mean_loss_regression = 0.5
#     std_loss_regression = 0.1
# 
#     # Define reference value for regression
#     reference_value_regression = 0.5
# 
#     # Calculate loss-weight update factor for regression
#     update_factor_regression = np.exp((mean_loss_regression - reference_value_regression) / std_loss_regression)
# 
#     # Normalize the update factor to ensure it is between 0 and 1
#     update_factor_regression = max(0.0, min(1.0, update_factor_regression))
# 
#     # Convert regression_weight to torch tensor before performing multiplication
#     regression_weight = torch.tensor(regression_weight, dtype=torch.float32, device=device)
# 
#     # Check if the update factor is close to 0 and the initial regression weights are all zeros
#     if update_factor_regression < 1e-6 and torch.all(regression_weight == 0):
#         # In this case, set the update factor to a small value to ensure some weight update occurs
#         update_factor_regression = 1e-3
# 
#     # Update the regression weight based on the calculated update factor
#     regression_weight *= update_factor_regression
# 
#     # Save the updated regression weight back to the separate directory
#     save_weights_to_directory(output_directory, regression_weight.cpu().numpy())
# 
#     # Print the updated regression weight after each epoch
#   # Print the updated regression weight after each epoch
# with np.printoptions(threshold=np.inf):
#     print(f"Regression Weight (Epoch {epoch + 1}): \n{regression_weight}")
# 
# 
# # End of the training loop for regression task
# 
# # ... (rest of your code for other tasks such as classification, segmentation, and survival)
# 
# # Now, you can copy and paste the code for updating weights and the training loop for other tasks as needed.

# === AFTER (edited) ===
import os
import numpy as np
import torch


def load_loss_weights_from_directory(directory_path):
    weight_files = [filename for filename in os.listdir(directory_path) if filename.endswith(".npy")]
    weight_files.sort()
    weights = []
    for filename in weight_files:
        arr = np.load(os.path.join(directory_path, filename))
        # Ensure scalar (0-D) arrays can be concatenated safely
        weights.append(np.atleast_1d(arr))

    if not weights:
        return np.array([], dtype=np.float32)

    return np.concatenate(weights)


def save_weights_to_directory(directory_path, weights):
    os.makedirs(directory_path, exist_ok=True)
    np.save(os.path.join(directory_path, "updated_regression_weights.npy"), weights)


regression_weights_directory = 'data/adjusted_survival_2019'


regression_weight = load_loss_weights_from_directory(regression_weights_directory)


num_epochs_update_regression = 5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


output_directory = 'data/updated_regression_weights'
os.makedirs(output_directory, exist_ok=True)

for epoch in range(num_epochs_update_regression):

    mean_loss_regression = 0.5
    std_loss_regression = 0.1


    reference_value_regression = 0.5


    update_factor_regression = np.exp((mean_loss_regression - reference_value_regression) / std_loss_regression)


    update_factor_regression = max(0.0, min(1.0, update_factor_regression))


    regression_weight = torch.tensor(regression_weight, dtype=torch.float32, device=device)


    if update_factor_regression < 1e-6 and torch.all(regression_weight == 0):

        update_factor_regression = 1e-3


    regression_weight *= update_factor_regression


    save_weights_to_directory(output_directory, regression_weight.cpu().numpy())



with np.printoptions(threshold=np.inf):
    print(f"Regression Weight (Epoch {epoch + 1}): \n{regression_weight}")