from base_nnm import NNAeroG

# --- AOD Model Instantiation ---

# Define the input dimension for the AOD model
aod_input_dim = 21

# Define the hidden layer architecture (using default values)
# This can be customized, e.g., hidden_dims=
default_hidden_dims = [256, 128, 64]
default_dropout_p = 0.3

# Create an instance of the NNAeroG model for AOD retrieval
aod_model = NNAeroG(
    input_dim=aod_input_dim,
    hidden_dims=default_hidden_dims,
    dropout_p=default_dropout_p
)

# Print the model architecture to verify
print("--- NNAeroG Model for AOD Retrieval ---")
print(f"Input Dimension (N): {aod_input_dim}")
print(aod_model)