# models.py

import torch
import torch.nn as nn

class CNNModel(nn.Module):
    """
    2D CNN for EEG-based emotion recognition using DE features.
    DE matrix shape: (62, 5 ) where 62 is the number of channels and 5 is frequency bands (delta, theta, alpha, beta, gamma).

    Input shape:  (batch_size, 1, 62, 5) 
    Output shape: (batch_size, num_classes) where num_classes=3 for positive, neutral, negative)
    """

    def __init__(self, config):
        super().__init__()

        self.config = config

        in_channels = config["dataset"]["conv_in_channels"] # 1 channel for Conv2d
        num_classes = config["dataset"]["num_classes"] # 3 classes
        eeg_channels = config["dataset"]["eeg_channels"]
        freq_bands = config["dataset"]["freq_bands"]

        # Convolutional layers for feature extraction
        self.conv_layers = nn.Sequential(
            # block 1 for extracting lower-level features 1 ->32
            nn.Conv2d(in_channels, 32, kernel_size=(3, 3), padding=1), # choosing kernel size of (3, 3) to capture 2d spatial patterns based on the size of the input DE matrix (62, 5)
            nn.BatchNorm2d(32), # batch normalization after convolutional layer to stabilize training and improve convergence
            nn.ReLU(),
            
            # block 2 for extracting higher-level features 32 -> 64
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1), # same kernel size for second conv layer to further capture spatial patterns while maintaining the spatial dimensions of the feature maps
            nn.BatchNorm2d(64),
            nn.ReLU(),  # ouput shape after conv layers: (batch_size, 64, 62, 5)
        )
        
        # Fully connected layer for classification
        self.fc_layers =nn.Sequential(
            nn.Linear(64 * 62 * 5, 128), # forward pass will flatten the feature maps from conv layers (64 channels, 62 height, 5 width) into a single vector of size 64*62*5=19840 for each sample, and then pass through a fully connected layer to reduce dimensionality to 128, I chose 128 as a common hidden layer size that balances model capacity and computational efficiency
            nn.ReLU(),
            nn.Dropout(0.5), # dropout rate of 0.5 to prevent overfitting by randomly setting half of the activations to zero during training, which encourages the model to learn more robust features that generalize better to unseen data
            nn.Linear(128, num_classes) # every batch will ouput a 128-dimensional feature vector with 3 output values corresponding to the 3 classes (positive, neutral, negative)
        )

    def forward(self, x):
        """
        Forward pass

        x: (batch_size, 1, 62, 5) input DE matrix with 1 channel(DE feature), 62 channels, and 5 frequency bands
        """

        assert x.ndim == 4, f"Expected input shape (B, 1, 62, 5), got {x.shape}" # sanity check for input shape
        x = self.conv_layers(x) # pass input through convolutional layers, output shape: (batch_size, 64, 62, 5)
        x = x.view(x.size(0), -1) # flatten the feature maps into a single vector for each sample, output shape: (batch_size, 64*62*5)
        x = self.fc_layers(x) # pass through fully connected layers for classification, output shape: (batch_size, num_classes)
        return x
    

class MLPClassifier(nn.Module):
    """
    MLP baseline for EEG emotion classification.

    Input shape:
        (B, 62, 5)

    Processing:
        Flatten (62 * 5 = 310)
        Fully connected layers

    Output shape:
        (B, 3)
    """

    def __init__(self, config):
        super().__init__()

        input_dim = (
            config["dataset"]["eeg_channels"]
            *
            config["dataset"]["freq_bands"]
        )

        num_classes = config["dataset"]["num_classes"]

        self.net = nn.Sequential(
            nn.Flatten(),

            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)

class MLPPlusClassifier(nn.Module):
    """
    Improved MLP for EEG emotion classification. Compared with the basic MLP, this version adds:
    - BatchNorm1d to stabilize hidden activations
    - Moderate Dropout for regularization
    - Intended to be trained with AdamW + weight decay

    Input shape:
        (B, 62, 5)

    Output shape:
        (B, 3)
    """

    def __init__(self, config):
        super().__init__()

        input_dim = (
            config["dataset"]["eeg_channels"]
            * config["dataset"]["freq_bands"]
        )
        num_classes = config["dataset"]["num_classes"]

        self.net = nn.Sequential(
            nn.Flatten(),

            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)


# get model function to initialize the model based on config    
def get_model(config):
    mode = config["mode"]
    model_type = config["model"]["type"]

    if mode =='de':
        if model_type == "cnn":
            return CNNModel(config)
        elif model_type == "mlp":
            return MLPClassifier(config)
        elif model_type == "mlp_plus":
            return MLPPlusClassifier(config)
    elif mode == 'raw':
        if model_type == "cnn":
            return CNNModelRaw(config)
        elif model_type == "mlp":
            return MLPClassifierRaw(config)
        elif model_type == "mlp_plus":
            return MLPPlusClassifier(config)
    
    raise NotImplementedError(f"Model type {model_type} for mode {mode} is not implemented.")
