# models.py

import torch
import torch.nn as nn
import torch.nn.functional as F

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
            nn.AdaptiveAvgPool2d((4, 5)), # add adaptive pooling to reduce the feature map size to (4, 5) for each channel, which helps reduce the number of parameters in the fully connected layers and prevents overfitting
        )
        
        # Fully connected layer for classification
        self.fc_layers =nn.Sequential(
            nn.Linear(64 * 4 * 5, 128), # forward pass will flatten the feature maps from conv layers (64 channels, 4 height, 5 width) into a single vector of size 64*4*5=1280 for each sample, and then pass through a fully connected layer to reduce dimensionality to 128, I chose 128 as a common hidden layer size that balances model capacity and computational efficiency
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
    
class CNNModelRaw(nn.Module):
    """
    2D CNN for EEG-based emotion recognition using raw EEG signals.
    Raw signal shape: (62, 800) where 62 is the number of channels and 800 is
    time samples per 4-second window (200Hz * 4s).

    Input shape:  (batch_size, 1, 62, 800)
    Output shape: (batch_size, num_classes) where num_classes=3 for positive, neutral, negative
    """

    def __init__(self, config):
        super().__init__()

        self.config = config

        in_channels = config["dataset"]["conv_in_channels"]  # 1 channel for Conv2d
        num_classes = config["dataset"]["num_classes"]       # 3 classes
        eeg_channels = config["dataset"]["eeg_channels"]      # 62
        window_size = config["dataset"]["window_size"]        # 800

        # Convolutional layers with pooling to progressively downsample the
        # time axis (800 is too long to flatten directly like the DE model does)
        self.conv_layers = nn.Sequential(
            # block 1: 1 -> 32, pool time axis by 4
            nn.Conv2d(in_channels, 32, kernel_size=(3, 7), padding=(1, 3)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 4)),  # only pool time axis, keep channel axis

            # block 2: 32 -> 64, pool time axis by 4 again
            nn.Conv2d(32, 64, kernel_size=(3, 7), padding=(1, 3)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 4)),

            nn.AdaptiveAvgPool2d((1, 1))  # add adaptive pooling to reduce feature map, (B,64,62,50) → (B,64,1,1)
        )

        # figure out the flattened feature size dynamically instead of hardcoding it,
        # so this doesn't break if window_size or eeg_channels ever change
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, eeg_channels, window_size)
            flat_dim = self.conv_layers(dummy).view(1, -1).shape[1]

        self.fc_layers = nn.Sequential(
            nn.Linear(flat_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        """
        Forward pass

        x: (batch_size, 1, 62, 800) input raw EEG window
        """
        assert x.ndim == 4, f"Expected input shape (B, 1, 62, 800), got {x.shape}"
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x
    

class CNNModelRaw1D(nn.Module):
    """
    1D CNN for EEG-based emotion recognition using raw EEG signals.
    Raw signal shape: (62, 800) where 62 is the number of channels and 800 is
    time samples per 4-second window (200Hz * 4s).

    Input shape:  (batch_size, 62, 800)
    Output shape: (batch_size, num_classes) where num_classes=3 for positive, neutral, negative
    """

    def __init__(self, config):
        super().__init__()

        self.config = config

        in_channels = config["dataset"]["eeg_channels"]  # 62 channels for Conv1d
        num_classes = config["dataset"]["num_classes"]   # 3 classes
        window_size = config["dataset"]["window_size"]   # 800

        # Convolutional layers with pooling to progressively downsample the
        # time axis (800 is too long to flatten directly)
        self.conv_layers = nn.Sequential(
            # block 1: 62 -> 64, pool time axis by 4
            nn.Conv1d(in_channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4),  # pool time axis

            # block 2: 64 -> 128, pool time axis by 4 again
            nn.Conv1d(64, 128, kernel_size=7, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4),

            nn.AdaptiveAvgPool1d(1)  # reduce to (B,128,1)
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        """
        Forward pass

        x: should be (batch_size, 62, 800) input raw EEG window
        """
        if x.ndim ==4:
            x = x.squeeze(1)  # remove channel dimension if present, (B, 1, 62, 800) -> (B, 62, 800)
        assert x.ndim == 3, f"Expected input shape (B, 62, 800), got {x.shape}"
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x
    

class GraphConvolution(nn.Module):
    """
    1D Graph Convolution Layer for EEG data: can learn spatial relationships between EEG channels based on the adjacency matrix of the graph.
    Input shape: (batch_size, num_channels, num_features), which is (batch_size, 62, 5) for DE features.
    Output shape: (batch_size, num_channels, out_features), which is (batch_size, 62, out_features) after graph convolution.
    """
    def __init__(self, in_features, out_features, bias=True): 
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features)) # learnable weight matrix of shape (in_features, out_features) to transform input features to output features
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features)) # learnable bias vector of shape (out_features) to be added to the output features after graph convolution
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight) # initialize the weight matrix using Xavier uniform initialization to ensure that the weights are scaled appropriately for the input and output dimensions
        if self.bias is not None:
            nn.init.zeros_(self.bias) # initialize the bias vector to zeros
    
    def forward(self, x, adj):
        """
        Forward pass of the graph convolution layer.
        x: input features of shape (batch_size, num_channels, in_features)
        adj: adjacency matrix of shape (num_channels, num_channels)(62, 62) representing the graph structure
        """
        support = torch.matmul(x, self.weight) # linear transformation of input features using the weight matrix
        output = torch.matmul(adj, support) # graph convolution operation using the adjacency matrix to aggregate information from neighboring nodes
        if self.bias is not None:
            output += self.bias # add bias to the output features if bias is enabled
        return output # shape of output: (batch_size, num_channels, out_features)
    
class DGCNN(nn.Module):
    """
    DGCNN model for EEG emotion recognition using DE features.
    Input shape: (batch_size, num_channels, num_features), which is (batch_size, 62, 5) for DE features.
    Output shape: (batch_size, num_classes), which is (batch_size, 3) for 3 emotion classes.
    Reference: Song et al., "EEG Emotion Recognition Using Dynamical GCN", IEEE TAC 2018.
    Main idea: use a learnable adjacency matrix to capture the dynamic relationships between EEG channels, and apply graph convolution to learn spatial features for emotion classification.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

        num_nodes = config["dataset"]["eeg_channels"] # number of EEG channels (62)
        in_features = config["dataset"]["freq_bands"] # number of frequency bands (5)
        num_classes = config["dataset"]["num_classes"] # number of emotion classes (3)

        # learnable adjacency matrix for (62 x 62) graph structure
        self.adj = nn.Parameter(torch.FloatTensor(num_nodes, num_nodes)) 
        nn.init.xavier_uniform_(self.adj) # initialize the adjacency matrix using Xavier uniform initialization to ensure that the weights are scaled appropriately for the input and output dimensions

        # 2 layer graph convolution layers, expand feature dimensions: 5 -> 32 -> 64
        self.gc1 = GraphConvolution(in_features, 32) # first graph convolution layer with output features of size 32
        self.gc2 = GraphConvolution(32, 64) # second graph convolution layer with output features of size 64
        
        # batch normalization and dropout in node dimension (num_channels) to stabilize training and improve generalization
        self.bn1 = nn.BatchNorm1d(num_nodes) # batch normalization for first graph after gc1
        self.bn2 = nn.BatchNorm1d(num_nodes) # batch normalization for second graph after gc2

        self.dropout = nn.Dropout(0.5) 

        # after graph convolution layer, node dimension with 64 features will be flattened to a single vector of size 62*64=3968 for each sample, and then passed through a fully connected layer to reduce dimensionality to 128, and finally output 3 classes 
        self.fc = nn.Sequential(
            nn.Linear(64 * num_nodes, 128), # fully connected layer to reduce dimensionality from 3968 to 128
            nn.ReLU(), 
            nn.Dropout(0.5), # dropout for regularization
            nn.Linear(128, num_classes) # final output layer for classification into 3 classes
        )
    
    def normalize_adj(self, adj):
        """
        Normalize the adjacency matrix using symmetric normalization: D^(-1/2) * A * D^(-1/2)
        where D is the degree matrix of A.
        This normalization ensures that the graph convolution operation is stable and prevents numerical issues during training.
        Input shape: adj(62, 62) which is the learnable adjacency matrix representing the graph structure of EEG channels.
        Output shape: (62, 62) normalized adjacency matrix for graph convolution.
        """
        # add self-loops to the adjacency matrix
        adj = adj + torch.eye(adj.size(0), device=adj.device) # add identity matrix to include self-connections
        adj = F.relu(adj) # apply ReLU to ensure non-negative values in the adjacency matrix
        degree = adj.sum(dim=1) # (62,) compute the degree of each node by summing the adjacency matrix along rows
        d_inv_sqrt = torch.pow(degree + 1e-8, -0.5)  # compute D^(-1/2)
        d_mat = torch.diag(d_inv_sqrt)  # create diagonal matrix D^(-1/2)

        return torch.matmul(torch.matmul(d_mat, adj), d_mat)  # return normalized adjacency matrix(62,62)
    
    def forward(self, x):
        """
        Input shape: (B, 62, 5)
        Output shape: (B, 3)
        """
        if x.ndim == 4:
            x = x.squeeze(1)  # remove the channel dimension if present, resulting in shape (B, 62, 5)
        assert x.ndim == 3, f"Expected (B,62,5), got {x.shape}"   # x: (B, 62, 5)

        adj = self.normalize_adj(self.adj) # normalize the learnable adjacency matrix for graph convolution

        # first graph convolution layer
        x = self.gc1(x, adj) #(B, 62, 5) -> (B, 62, 32)
        x = self.bn1(x) #(B, 62, 32)
        x = F.relu(x) #(B, 62, 32)
        x = self.dropout(x) #(B, 62, 32)

        # second graph convolution layer
        x = self.gc2(x, adj) #(B, 62, 32) -> (B, 62, 64)
        x = self.bn2(x) #(B, 62, 64)
        x = F.relu(x) #(B, 62, 64)
        
        # flatten all feature nodes  -->classification
        x = x.view(x.size(0), -1) #(B, 62, 63) --> (B, 62*64=3968)
        x = self.fc(x) #(B, 3968) -> (B, 128) -> (B, 3)
        return x   # (B, 3)
    

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
            * config["dataset"]["freq_bands"]
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


class MLPClassifierRaw(nn.Module):
    """
    MLP baseline for raw EEG windows.

    Input:
        (B, 1, 62, 800)

    Output:
        (B, 3)
    """

    def __init__(self, config):
        super().__init__()

        input_dim = (
            config["dataset"]["eeg_channels"]
            * config["dataset"]["window_size"]
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

class LSTMClassifierRaw(nn.Module):
    """
    Input shape:
        (bs, 1, 62, 800)

    LSTM input shape after reshaping:
        (bs, 800, 62)

    Output shape:
        (bs, num_classes), where num_classes=3
    """
    def __init__(self, config):
        super().__init__()

        input_size = config["dataset"]["eeg_channels"]
        hidden_size = 64
        num_layers = 1
        num_classes = config["dataset"]["num_classes"]

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.0
        )
        # bidirectional=True gives hidden_size * 2
        # mean pooling and max pooling are concatenated, so classifier input is hidden_size * 4
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        assert x.ndim == 4, f"Expected input shape (B, 1, 62, 800), got {x.shape}"

        x = x.squeeze(1)          # (B, 62, 800)
        x = x.transpose(1, 2)     # (B, 800, 62)

        out, _ = self.lstm(x)     # (B, 800, hidden_size * 2)

        mean_pool = out.mean(dim=1)       # (B, hidden_size * 2)
        max_pool, _ = out.max(dim=1)      # (B, hidden_size * 2)

        feat = torch.cat([mean_pool, max_pool], dim=1)   # (B, hidden_size * 4)

        return self.classifier(feat)


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
        elif model_type == "dgcnn":
            return DGCNN(config)
    elif mode == 'raw':
        if model_type == "cnn":
            return CNNModelRaw(config)
        elif model_type == "cnn1d":
            return CNNModelRaw1D(config)
        elif model_type == "mlp":
            return MLPClassifierRaw(config)
        elif model_type == "lstm":
            return LSTMClassifierRaw(config)
    
    raise NotImplementedError(f"Model type {model_type} for mode {mode} is not implemented.")
