import torch
import torch.nn as nn
from typing import List

class NNAeroG(nn.Module):
    """
    A configurable PyTorch implementation of the Neural Network AEROsol
    Retrieval for Geostationary Satellite (NNAeroG) model architecture.

    This class builds a Fully Connected Neural Network (FCNN) based on the
    architecture described by Chen et al. (2022). The network consists of an
    input layer, a series of hidden blocks, and a final output layer. Each
    hidden block is composed of a sequence of:
    1. Fully Connected (Linear) layer
    2. ReLU activation function
    3. Batch Normalization layer
    4. Dropout layer

    The architecture is flexible and can be configured by specifying the
    number of input features, the dimensions of the hidden layers, and the
    dropout probability.

    Reference:
    Chen, X., et al. (2022). Neural Network AEROsol Retrieval for
    Geostationary Satellite (NNAeroG) Based on Temporal, Spatial and
    Spectral Measurements. Remote Sensing, 14(4), 980.
    """
    def __init__(self,
                 input_dim: int,
                 hidden_dims: List[int],
                 dropout_p: float = 0.3):
        """
        Initializes the NNAeroG model.

        Args:
            input_dim (int): The number of features in the input vector (N).
                             This corresponds to the values in Table 2 of the
                             source paper (e.g., 14 for AOD, 475 for AE).
            hidden_dims (List[int]): A list of integers where each integer
                                     specifies the number of neurons in a
                                     hidden layer. The length of the list
                                     determines the number of hidden layers.
            dropout_p (float, optional): The probability for the Dropout
                                         layers. Defaults to 0.3.
        """
        super(NNAeroG, self).__init__()

        if not hidden_dims:
            raise ValueError("hidden_dims list cannot be empty.")

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_p = dropout_p

        # Create a list to hold all the layers of the network
        layers = []
        
        # Define the first hidden layer connected to the input
        current_dim = self.input_dim
        
        # Dynamically build the hidden layers
        for h_dim in self.hidden_dims:
            # Full Connected (Linear) layer
            layers.append(nn.Linear(current_dim, h_dim))
            # ReLU activation
            layers.append(nn.ReLU(inplace=True))
            # Batch Normalization
            layers.append(nn.BatchNorm1d(h_dim))
            # Dropout for regularization
            layers.append(nn.Dropout(p=self.dropout_p))
            current_dim = h_dim

        # The main network body composed of the sequential hidden blocks
        self.hidden_layers = nn.Sequential(*layers)

        # Final output layer that produces a single regression value
        self.output_layer = nn.Linear(current_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the model.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, 1),
                          representing the predicted aerosol parameter.
        """
        # Pass input through the sequence of hidden layers
        x = self.hidden_layers(x)
        # Pass through the final output layer to get the prediction
        output = self.output_layer(x)
        return output